"""Mock joke agent with guardrails for testing.

This agent uses create_agent() with guardrails to test that guardrails are properly invoked.
"""

from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentBuiltInValidatorGuardrail,
    AgentCustomGuardrail,
    AgentEscalationRecipientType,
    AgentGuardrailBlockAction,
    AgentGuardrailEscalateAction,
    AgentGuardrailEscalateActionApp,
    AgentGuardrailFilterAction,
    AgentWordOperator,
    AgentWordRule,
    StandardRecipient,
)
from uipath.core.guardrails.guardrails import FieldReference, FieldSource
from uipath.platform.guardrails.guardrails import NumberParameterValue

from uipath_langchain.chat.models import UiPathChat


# Mock Sentence Analyzer Tool
class SentenceAnalyzerTool(BaseTool):
    """Mock tool that analyzes sentences."""

    name: str = "Agent___Sentence_Analyzer"  # Use sanitized name (no spaces)
    description: str = "Analyzes a sentence and provides insights about its structure"

    def _run(self, sentence: str) -> str:
        """Synchronous execution."""
        # Simple mock analysis - return structured output including the input phrase
        word_count = len(sentence.split())
        char_count = len(sentence)

        # Return a JSON string with both analysis and the input phrase
        import json

        result = {
            "analysis": f"Analysis: {word_count} words, {char_count} characters. Sentence structure is valid.",
            "input_phrase": sentence,  # Include the input phrase in the output
        }
        return json.dumps(result)

    async def _arun(self, sentence: str) -> str:
        """Asynchronous execution."""
        return self._run(sentence)


# Input/Output Models
class AgentInput(BaseModel):
    """Input schema for the joke agent."""

    word: str = Field(..., description="The word to base the joke on")


class AgentOutput(BaseModel):
    """Output schema for the joke agent."""

    joke: str = Field(..., description="The generated family-friendly joke")
    randomName: str = Field(..., description="A randomly generated name")
    analysis: str = Field(
        ..., description="The analysis result from the SentenceAnalyzer tool"
    )


# Create tools
sentence_analyzer_tool = SentenceAnalyzerTool()
all_tools = [sentence_analyzer_tool]

# Create LLM (will be mocked in tests)
llm = UiPathChat(
    model="gpt-4o-2024-11-20",
    temperature=0.0,
    max_tokens=500,
)


# Agent Messages Function
def create_messages(state) -> Sequence[SystemMessage | HumanMessage]:
    """Create messages for the agent based on input state."""
    import os

    # Note: Due to guardrails subgraph using AgentGuardrailsGraphState,
    # the input fields are not available in the state when guardrails are present.
    # For testing, we use an environment variable to inject test-specific prompts.
    test_prompt = os.environ.get("TEST_PROMPT_INJECTION")

    if test_prompt:
        # Use the test prompt directly (for prompt injection testing)
        word = test_prompt
    else:
        # Try to get word from state, fall back to "test"
        word = getattr(state, "word", "test")

    system_prompt = """You are a joke generator assistant.
Generate a family-friendly joke based on the given word, create a random name,
and use the SentenceAnalyzer tool to analyze the combined text."""

    user_prompt = f"""Generate a family-friendly joke about: {word}
Then create a random name and use the SentenceAnalyzer tool to analyze the joke and name together."""

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


# Define guardrails programmatically
# Naming convention: {action}_{scope}_{type}_guardrail
filter_tool_custom_guardrail = AgentCustomGuardrail(
    guardrail_type="custom",
    id="0dac2299-a8ae-43aa-8703-3eb93c657b2a",
    name="Guardrail on output for donkey",
    description="Filters out the word 'donkey' from tool output field 'input_phrase'",
    enabled_for_evals=True,
    selector={
        "scopes": ["Tool"],
        "matchNames": [sentence_analyzer_tool.name],  # Use the actual tool name
    },
    rules=[
        AgentWordRule(
            rule_type="word",
            field_selector={
                "selector_type": "specific",
                "fields": [{"path": "input_phrase", "source": "output"}],
            },
            operator=AgentWordOperator.CONTAINS,
            value="donkey",
        )
    ],
    action=AgentGuardrailFilterAction(
        action_type="filter",
        fields=[FieldReference(path="input_phrase", source=FieldSource.OUTPUT)],
    ),
)

# PII Detection Guardrail at AGENT level
block_agent_pii_detection_guardrail = AgentBuiltInValidatorGuardrail(
    guardrail_type="builtInValidator",
    id="3b4d5416-202a-47ab-bba6-89fa8940a5cf",
    name="PII detection guardrail",
    description="Detects personally identifiable information using Azure Cognitive Services",
    validator_type="pii_detection",
    validator_parameters=[
        {
            "parameter_type": "enum-list",
            "id": "entities",
            "value": ["Email", "Address", "Person"],
        },
        {
            "parameter_type": "map-enum",
            "id": "entityThresholds",
            "value": {
                "Email": 0.5,
                "Address": 0.5,
                "Person": 0.5,
            },
        },
    ],
    action=AgentGuardrailBlockAction(
        action_type="block",
        reason="PII detected in agent input/output",
    ),
    enabled_for_evals=True,
    selector={
        "scopes": ["Agent"],  # AGENT level guardrail
        "matchNames": [],
    },
)

# Prompt Injection Guardrail at LLM level
block_llm_prompt_injection_guardrail = AgentBuiltInValidatorGuardrail(
    guardrail_type="builtInValidator",
    id="255b1220-97f8-4d79-be8e-052a664b2b90",
    name="Prompt injection guardrail",
    description="Detects malicious attack attempts (e.g. prompt injection, jailbreak) in LLM calls",
    validator_type="prompt_injection",
    validator_parameters=[
        NumberParameterValue(
            parameter_type="number",
            id="threshold",
            value=0.5,
        ),
    ],
    action=AgentGuardrailBlockAction(
        action_type="block",
        reason="Prompt injection detected",
    ),
    enabled_for_evals=True,
    selector={
        "scopes": ["Llm"],  # LLM level guardrail
        "matchNames": [],
    },
)

# Block Guardrail for Tool - blocks if input matches forbidden pattern
block_tool_forbidden_pattern_guardrail = AgentCustomGuardrail(
    guardrail_type="custom",
    id="block-forbidden-pattern-123",
    name="Block forbidden pattern in tool input",
    description="Blocks tool execution if input contains forbidden words",
    enabled_for_evals=True,
    selector={
        "scopes": ["Tool"],
        "matchNames": [sentence_analyzer_tool.name],
    },
    rules=[
        AgentWordRule(
            rule_type="word",
            field_selector={
                "selector_type": "specific",
                "fields": [{"path": "sentence", "source": "input"}],
            },
            operator=AgentWordOperator.CONTAINS,
            value="forbidden",  # Block if input contains "forbidden"
        )
    ],
    action=AgentGuardrailBlockAction(
        action_type="block",
        reason="Tool execution blocked due to forbidden pattern in input",
    ),
)

# Block Guardrail for Tool - blocks if input matches regex pattern (MATCHES_REGEX version)
# This guardrail uses MATCHES_REGEX operator to test regex pattern matching
block_tool_forbidden_pattern_regex_guardrail = AgentCustomGuardrail(
    guardrail_type="custom",
    id="block-forbidden-pattern-regex-123",
    name="Block forbidden pattern in tool input (regex)",
    description="Blocks tool execution if input matches forbidden regex pattern",
    enabled_for_evals=True,
    selector={
        "scopes": ["Tool"],
        "matchNames": [sentence_analyzer_tool.name],
    },
    rules=[
        AgentWordRule(
            rule_type="word",
            field_selector={
                "selector_type": "specific",
                "fields": [{"path": "sentence", "source": "input"}],
            },
            operator=AgentWordOperator.MATCHES_REGEX,
            value=r".*forbidden.*",
        )
    ],
    action=AgentGuardrailBlockAction(
        action_type="block",
        reason="Tool execution blocked due to forbidden pattern in input",
    ),
)

# PII Detection Guardrail at TOOL level - blocks if tool input contains PII
block_tool_pii_detection_guardrail = AgentBuiltInValidatorGuardrail(
    guardrail_type="builtInValidator",
    id="tool-pii-detection-456",
    name="Tool PII detection guardrail",
    description="Detects PII in tool inputs",
    validator_type="pii_detection",
    validator_parameters=[
        {
            "parameter_type": "enum-list",
            "id": "entities",
            "value": ["Email", "Address", "Person"],
        },
        {
            "parameter_type": "map-enum",
            "id": "entityThresholds",
            "value": {
                "Email": 0.5,
                "Address": 0.5,
                "Person": 0.5,
            },
        },
    ],
    action=AgentGuardrailBlockAction(
        action_type="block",
        reason="PII detected in tool input",
    ),
    enabled_for_evals=True,
    selector={
        "scopes": ["Tool"],  # TOOL level guardrail
        "matchNames": [sentence_analyzer_tool.name],
    },
)

# PII Detection Guardrail at LLM level with HITL Escalation
escalate_llm_pii_detection_guardrail = AgentBuiltInValidatorGuardrail(
    guardrail_type="builtInValidator",
    id="llm-pii-escalation-789",
    name="LLM PII escalation guardrail",
    description="Escalates to human when PII is detected in LLM output",
    validator_type="pii_detection",
    validator_parameters=[
        {
            "parameter_type": "enum-list",
            "id": "entities",
            "value": ["Email", "Address", "Person"],
        },
        {
            "parameter_type": "map-enum",
            "id": "entityThresholds",
            "value": {
                "Email": 0.5,
                "Address": 0.5,
                "Person": 0.5,
            },
        },
    ],
    action=AgentGuardrailEscalateAction(
        action_type="escalate",
        app=AgentGuardrailEscalateActionApp(
            name="ReviewPII",
            folder_name="/Test/Guardrails",
            version=1,
        ),
        recipient=StandardRecipient(
            type=AgentEscalationRecipientType.USER_EMAIL,
            value="admin@test.com",
            display_name="Admin",
        ),
    ),
    enabled_for_evals=True,
    selector={
        "scopes": ["Llm"],  # LLM level guardrail
        "matchNames": [],
    },
)
