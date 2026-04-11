"""Middleware-based joke agent for guardrails parity testing.

This agent implements the same guardrail configuration as parity_agent_decorator.py
using the UiPath middleware API (create_agent(middleware=[...])). Both agents are
used in test_guardrails_parity.py to verify 1:1 behavioral parity between the two
guardrail flavors.

Guardrails configured:
- "Agent PII Detection"        — AGENT scope, PII (PERSON), PRE, BlockAction
- "LLM Prompt Injection Detection" — LLM scope, Prompt Injection, PRE, BlockAction
- "LLM PII Detection"          — LLM scope, PII (EMAIL), PRE, LogAction(WARNING)
- "Tool PII Detection"         — TOOL scope, PII (EMAIL, PHONE), PRE, LogAction(WARNING)
- "Tool PII Block Detection"   — TOOL scope, PII (PERSON), PRE, BlockAction
- "Joke Content Word Filter"   — TOOL scope, Deterministic, PRE, CustomFilterAction
- "Joke Content Length Limiter"— TOOL scope, Deterministic, PRE, BlockAction
- "Joke Content Always Filter" — TOOL scope, Deterministic (empty), POST, CustomFilterAction
"""

import re
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)

from uipath_langchain.chat.openai import UiPathChatOpenAI
from uipath_langchain.guardrails import (
    BlockAction,
    GuardrailAction,
    GuardrailExecutionStage,
    LogAction,
    PIIDetectionEntity,
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
)
from uipath_langchain.guardrails.actions import LoggingSeverityLevel
from uipath_langchain.guardrails.enums import PIIDetectionEntityType

# ---------------------------------------------------------------------------
# Custom filter action (defined inline — no external middleware.py import)
# ---------------------------------------------------------------------------


@dataclass
class CustomFilterAction(GuardrailAction):
    """Filters/replaces a word in tool input when a violation is detected."""

    word_to_filter: str
    replacement: str = "***"

    def _filter(self, text: str) -> str:
        return re.sub(
            re.escape(self.word_to_filter), self.replacement, text, flags=re.IGNORECASE
        )

    def handle_validation_result(
        self,
        result: GuardrailValidationResult,
        data: str | dict[str, Any],
        guardrail_name: str,
    ) -> str | dict[str, Any] | None:
        if result.result != GuardrailValidationResultType.VALIDATION_FAILED:
            return None
        if isinstance(data, str):
            return self._filter(data)
        if isinstance(data, dict):
            filtered = data.copy()
            for key in ["joke", "text", "content", "message", "input", "output"]:
                if key in filtered and isinstance(filtered[key], str):
                    filtered[key] = self._filter(filtered[key])
            return filtered
        return data


# ---------------------------------------------------------------------------
# Input / Output schemas
# ---------------------------------------------------------------------------


class Input(BaseModel):
    """Input schema for the joke agent."""

    topic: str


class Output(BaseModel):
    """Output schema for the joke agent."""

    joke: str


# ---------------------------------------------------------------------------
# Tool — echoes input in output so filter tests can assert on ToolMessage content
# ---------------------------------------------------------------------------


@tool
def analyze_joke_syntax(joke: str) -> str:
    """Analyze the syntax of a joke by counting words and letters.

    Args:
        joke: The joke text to analyze

    Returns:
        A string with the analysis results including the input joke
    """
    words = joke.split()
    word_count = len(words)
    letter_count = sum(1 for char in joke if char.isalpha())
    # Include input so parity tests can verify filter modified it
    return f"Input: {joke}\nWords number: {word_count}\nLetters: {letter_count}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI assistant designed to generate family-friendly jokes.
1. Generate a family-friendly joke based on the given topic.
2. Use the analyze_joke_syntax tool to analyze the joke's syntax.
3. Ensure your output includes the joke.
Keep jokes appropriate for children, free from offensive language."""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = UiPathChatOpenAI(temperature=0.7, max_tokens=500, use_responses_api=True)


# ---------------------------------------------------------------------------
# Agent with full middleware guardrail stack
# ---------------------------------------------------------------------------

agent = create_agent(
    model=llm,
    tools=[analyze_joke_syntax],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        # AGENT scope PII — BlockAction
        *UiPathPIIDetectionMiddleware(
            name="Agent PII Detection",
            scopes=[GuardrailScope.AGENT],
            action=BlockAction(),
            entities=[PIIDetectionEntity(PIIDetectionEntityType.PERSON, 0.5)],
        ),
        # LLM scope Prompt Injection — BlockAction
        *UiPathPromptInjectionMiddleware(
            name="LLM Prompt Injection Detection",
            action=BlockAction(),
            threshold=0.5,
        ),
        # LLM scope PII — LogAction
        *UiPathPIIDetectionMiddleware(
            name="LLM PII Detection",
            scopes=[GuardrailScope.LLM],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
        ),
        # Tool scope PII — LogAction (email + phone)
        *UiPathPIIDetectionMiddleware(
            name="Tool PII Detection",
            scopes=[GuardrailScope.TOOL],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[
                PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.PHONE_NUMBER, 0.5),
            ],
            tools=[analyze_joke_syntax],
        ),
        # Tool scope PII — BlockAction (person name)
        *UiPathPIIDetectionMiddleware(
            name="Tool PII Block Detection",
            scopes=[GuardrailScope.TOOL],
            action=BlockAction(),
            entities=[PIIDetectionEntity(PIIDetectionEntityType.PERSON, 0.5)],
            tools=[analyze_joke_syntax],
        ),
        # Tool deterministic — filter "donkey" PRE
        *UiPathDeterministicGuardrailMiddleware(
            tools=[analyze_joke_syntax],
            rules=[lambda input_data: "donkey" in input_data.get("joke", "").lower()],
            action=CustomFilterAction(
                word_to_filter="donkey", replacement="[censored]"
            ),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Word Filter",
        ),
        # Tool deterministic — block length > 1000 PRE
        *UiPathDeterministicGuardrailMiddleware(
            tools=[analyze_joke_syntax],
            rules=[lambda input_data: len(input_data.get("joke", "")) > 1000],
            action=BlockAction(title="Joke too long", detail="Joke > 1000 chars"),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Length Limiter",
        ),
        # Tool deterministic — always-filter "words" POST
        *UiPathDeterministicGuardrailMiddleware(
            tools=[analyze_joke_syntax],
            rules=[],
            action=CustomFilterAction(word_to_filter="words", replacement="words++"),
            stage=GuardrailExecutionStage.POST,
            name="Joke Content Always Filter",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Wrapper graph node
# ---------------------------------------------------------------------------


async def joke_node(state: Input) -> Output:
    """Convert topic to messages, call agent, and extract joke."""
    messages = [
        HumanMessage(
            content=f"Generate a family-friendly joke based on the topic: {state.topic}"
        )
    ]
    result = await agent.ainvoke({"messages": messages})  # type: ignore[arg-type]
    joke = result["messages"][-1].content
    return Output(joke=joke)


# Build wrapper graph with custom input/output schemas
builder = StateGraph(Input, input_schema=Input, output_schema=Output)
builder.add_node("joke", joke_node)
builder.add_edge(START, "joke")
builder.add_edge("joke", END)

graph = builder.compile()
