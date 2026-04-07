"""Joke generating agent that creates family-friendly jokes based on a topic."""

import logging
import re
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)

from uipath_langchain.chat import UiPathChat
from uipath_langchain.guardrails import (
    BlockAction,
    CustomValidator,
    GuardrailAction,
    GuardrailExclude,
    GuardrailExecutionStage,
    HarmfulContentEntity,
    HarmfulContentValidator,
    IntellectualPropertyValidator,
    LogAction,
    LoggingSeverityLevel,
    PIIDetectionEntity,
    PIIValidator,
    UserPromptAttacksValidator,
    guardrail,
)
from uipath_langchain.guardrails.enums import (
    HarmfulContentEntityType,
    IntellectualPropertyEntityType,
    PIIDetectionEntityType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom filter action (defined locally)
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
            filtered = self._filter(data)
            print(
                f"[FILTER][{guardrail_name}] '{self.word_to_filter}' replaced → '{filtered[:80]}'"
            )
            return filtered
        if isinstance(data, dict):
            filtered_data = data.copy()
            for key in ["joke", "text", "content", "message", "input", "output"]:
                if key in filtered_data and isinstance(filtered_data[key], str):
                    filtered_data[key] = self._filter(filtered_data[key])
            print(f"[FILTER][{guardrail_name}] dict filtered")
            return filtered_data
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
# Reusable validators (declared once, used in multiple @guardrail decorators)
# ---------------------------------------------------------------------------

pii_email = PIIValidator(
    entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, threshold=0.5)],
)

pii_email_phone = PIIValidator(
    entities=[
        PIIDetectionEntity(PIIDetectionEntityType.EMAIL, threshold=0.5),
        PIIDetectionEntity(PIIDetectionEntityType.PHONE_NUMBER, threshold=0.5),
    ],
)


# ---------------------------------------------------------------------------
# Custom function with guardrails (pure Python — not a LangChain object)
# ---------------------------------------------------------------------------


@guardrail(
    validator=CustomValidator(lambda args: "donkey" in args.get("topic", "").lower()),
    action=CustomFilterAction(word_to_filter="donkey", replacement="[topic redacted]"),
    stage=GuardrailExecutionStage.PRE,
    name="Topic Word Filter",
)
def format_joke_for_display(
    topic: str,
    joke: str,
    config: Annotated[dict[str, Any], GuardrailExclude()],
) -> str:
    """Format a joke for display, combining topic and joke text.

    Args:
        topic: The joke topic (PRE guardrail checks for banned words).
        joke: The generated joke text.
        config: Display configuration — excluded from guardrail evaluation.

    Returns:
        A formatted string combining the prefix, topic, and joke.
    """
    prefix = config.get("prefix", "Here's your joke")
    return f"{prefix}!\nTopic: {topic}\nJoke: {joke}"


# ---------------------------------------------------------------------------
# LLM with guardrails (prompt injection + PII at LLM scope)
# ---------------------------------------------------------------------------


@guardrail(
    validator=UserPromptAttacksValidator(),
    action=BlockAction(),
    name="LLM User Prompt Attacks Detection",
    stage=GuardrailExecutionStage.PRE,
)
@guardrail(
    validator=IntellectualPropertyValidator(
        entities=[IntellectualPropertyEntityType.TEXT],
    ),
    action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
    name="LLM Intellectual Property Detection",
    stage=GuardrailExecutionStage.POST,
)
@guardrail(
    validator=pii_email,
    action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
    name="LLM PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
def create_llm():
    """Create LLM instance with guardrails."""
    return UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)


llm = create_llm()


# ---------------------------------------------------------------------------
# Tool with guardrails (deterministic + PII at TOOL scope)
# ---------------------------------------------------------------------------


@guardrail(
    validator=CustomValidator(lambda args: "donkey" in args.get("joke", "").lower()),
    action=CustomFilterAction(word_to_filter="donkey", replacement="[censored]"),
    stage=GuardrailExecutionStage.PRE,
    name="Joke Content Word Filter",
)
@guardrail(
    validator=CustomValidator(lambda args: len(args.get("joke", "")) > 1000),
    action=BlockAction(
        title="Joke is too long", detail="The generated joke is too long"
    ),
    stage=GuardrailExecutionStage.PRE,
    name="Joke Content Length Limiter",
)
@guardrail(
    validator=CustomValidator(lambda args: True),
    action=CustomFilterAction(word_to_filter="words", replacement="words++"),
    stage=GuardrailExecutionStage.POST,
    name="Joke Content Always Filter",
)
@guardrail(
    validator=pii_email_phone,
    action=LogAction(
        severity_level=LoggingSeverityLevel.WARNING,
        message="Email or phone number detected",
    ),
    name="Tool PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
@tool
def analyze_joke_syntax(joke: str) -> str:
    """Analyze the syntax of a joke by counting words and letters.

    Args:
        joke: The joke text to analyze

    Returns:
        A string with the analysis results showing word count and letter count
    """
    words = joke.split()
    word_count = len(words)
    letter_count = sum(1 for char in joke if char.isalpha())
    return f"Words number: {word_count}\nLetters: {letter_count}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI assistant designed to generate family-friendly jokes. Your process is as follows:

1. Generate a family-friendly joke based on the given topic.
2. Use the analyze_joke_syntax tool to analyze the joke's syntax (word count and letter count).
3. Ensure your output includes the joke.

When creating jokes, ensure they are:

1. Appropriate for children
2. Free from offensive language or themes
3. Clever and entertaining
4. Not based on stereotypes or sensitive topics

If you're unable to generate a suitable joke for any reason, politely explain why and offer to try again with a different topic.

Example joke: Topic: "banana" Joke: "Why did the banana go to the doctor? Because it wasn't peeling well!"

Remember to always include the 'joke' property in your output to match the required schema."""


# ---------------------------------------------------------------------------
# Agent with PII guardrail at AGENT scope
# ---------------------------------------------------------------------------


@guardrail(
    validator=HarmfulContentValidator(
        entities=[HarmfulContentEntity(HarmfulContentEntityType.VIOLENCE, threshold=2)],
    ),
    action=BlockAction(),
    name="Agent Harmful Content Detection",
    stage=GuardrailExecutionStage.PRE,
)
@guardrail(
    validator=PIIValidator(
        entities=[PIIDetectionEntity(PIIDetectionEntityType.PERSON, threshold=0.5)],
    ),
    action=BlockAction(
        title="Person name detection",
        detail="Person name detected and is not allowed",
    ),
    name="Agent PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
def create_joke_agent():
    """Create the joke agent with guardrails."""
    return create_agent(
        model=llm,
        tools=[analyze_joke_syntax],
        system_prompt=SYSTEM_PROMPT,
    )


agent = create_joke_agent()


# ---------------------------------------------------------------------------
# Wrapper graph node
# ---------------------------------------------------------------------------


@guardrail(
    validator=PIIValidator(
        entities=[PIIDetectionEntity(PIIDetectionEntityType.PERSON, threshold=0.5)],
    ),
    action=BlockAction(
        title="Person name detection in topic",
        detail="Person name detected in the node input and is not allowed",
    ),
    name="Node Input PII Detection",
    stage=GuardrailExecutionStage.PRE,
)
async def joke_node(state: Input) -> Output:
    """Convert topic to messages, call agent, and extract joke."""
    messages = [
        HumanMessage(
            content=f"Generate a family-friendly joke based on the topic: {state.topic}"
        )
    ]
    result = await agent.ainvoke({"messages": messages})
    joke = result["messages"][-1].content
    # format_joke_for_display is a plain Python function guarded by @guardrail:
    # - PRE collects {"topic": ..., "joke": ...} and checks for banned words
    # - "config" is excluded via GuardrailExclude and never sent to the guardrail
    display = format_joke_for_display(
        topic=state.topic,
        joke=joke,
        config={"prefix": "Here's your joke"},
    )
    logger.info("Display output: %s", display)
    return Output(joke=joke)


# Build wrapper graph with custom input/output schemas
builder = StateGraph(Input, input_schema=Input, output_schema=Output)
builder.add_node("joke", joke_node)
builder.add_edge(START, "joke")
builder.add_edge("joke", END)

graph = builder.compile()
