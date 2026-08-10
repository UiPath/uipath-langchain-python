"""Joke generating agent that creates family-friendly jokes based on a topic."""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from middleware import CustomFilterAction, LoggingMiddleware
from pydantic import BaseModel
from uipath.core.guardrails import GuardrailScope

from uipath_langchain.chat import UiPathChat
from uipath_langchain.guardrails import (
    BlockAction,
    EscalateAction,
    GuardrailExecutionStage,
    HarmfulContentEntity,
    LogAction,
    PIIDetectionEntity,
    UiPathDeterministicGuardrailMiddleware,
    UiPathHarmfulContentMiddleware,
    UiPathIntellectualPropertyMiddleware,
    UiPathLLMAsJudgeMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathUserPromptAttacksMiddleware,
)
from uipath_langchain.guardrails.actions import LoggingSeverityLevel
from uipath_langchain.guardrails.enums import (
    HarmfulContentEntityType,
    IntellectualPropertyEntityType,
    PIIDetectionEntityType,
)


# Define input schema for the agent
class Input(BaseModel):
    """Input schema for the joke agent."""

    topic: str


class Output(BaseModel):
    """Output schema for the joke agent."""

    joke: str


# Initialize UiPathChat LLM
llm = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)


@tool
def analyze_joke_syntax(joke: str) -> str:
    """Analyze the syntax of a joke by counting words and letters.

    Args:
        joke: The joke text to analyze

    Returns:
        A string with the analysis results showing word count and letter count
    """
    # Count words (split by whitespace)
    words = joke.split()
    word_count = len(words)

    # Count letters (only alphabetic characters, excluding spaces and punctuation)
    letter_count = sum(1 for char in joke if char.isalpha())

    return f"Words number: {word_count}\nLetters: {letter_count}"


# System prompt based on agent1.json
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

agent = create_agent(
    model=llm,
    tools=[analyze_joke_syntax],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        *LoggingMiddleware,
        # PII detection on the agent scope. On a violation it escalates to the
        # Guardrail Escalation Action App for human review via the documented
        # HITL interrupt(CreateEscalation(...)) — the run suspends until a human
        # approves (optionally editing the content) or rejects.
        *UiPathPIIDetectionMiddleware(
            name="PII escalation guardrail",
            scopes=[GuardrailScope.AGENT],
            # PRE only → validate the input once, so the escalation triggers a
            # single time per run (AGENT scope would otherwise check both
            # before_agent and after_agent).
            stage=GuardrailExecutionStage.PRE,
            action=EscalateAction(
                # Escalation Action App — declared as a binding in bindings.json
                # (resource "app"). Studio/deploy resolves and can override it;
                # locally these literal values are used.
                app_name="Guardrail.Escalation.Action.App.2",
                app_folder_path="Shared",
            ),
            entities=[
                PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.CREDIT_CARD_NUMBER, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.PHONE_NUMBER, 0.5),
            ],
        ),
        *UiPathPIIDetectionMiddleware(
            name="Tool PII detector",
            scopes=[GuardrailScope.TOOL],
            stage=GuardrailExecutionStage.PRE_AND_POST,
            action=BlockAction(),
            entities=[
                PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.CREDIT_CARD_NUMBER, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.PHONE_NUMBER, 0.5),
            ],
            tools=[analyze_joke_syntax],
            enabled_for_evals=False,
        ),
        *UiPathHarmfulContentMiddleware(
            name="Tool Harmful Content Detection",
            scopes=[GuardrailScope.TOOL],
            stage=GuardrailExecutionStage.PRE_AND_POST,
            action=BlockAction(),
            entities=[
                HarmfulContentEntity(HarmfulContentEntityType.VIOLENCE, threshold=2),
                HarmfulContentEntity(HarmfulContentEntityType.HATE, threshold=2),
            ],
            tools=[analyze_joke_syntax],
        ),
        *UiPathUserPromptAttacksMiddleware(
            name="User Prompt Attacks Detection",
            action=BlockAction(),
            enabled_for_evals=False,
        ),
        *UiPathHarmfulContentMiddleware(
            name="Harmful Content Detection",
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=BlockAction(),
            entities=[
                HarmfulContentEntity(HarmfulContentEntityType.VIOLENCE, threshold=2),
            ],
        ),
        *UiPathIntellectualPropertyMiddleware(
            name="Intellectual Property Detection",
            scopes=[GuardrailScope.LLM],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[IntellectualPropertyEntityType.TEXT],
        ),
        # Custom FilterAction example: demonstrates how developers can implement their own actions
        *UiPathDeterministicGuardrailMiddleware(
            tools=[analyze_joke_syntax],
            rules=[
                lambda input_data: "donkey" in input_data.get("joke", "").lower(),
            ],
            action=CustomFilterAction(
                word_to_filter="donkey",
                replacement="*",
            ),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Validator",
            enabled_for_evals=False,
        ),
        *UiPathDeterministicGuardrailMiddleware(
            tools=[analyze_joke_syntax],
            rules=[
                lambda input_data: len(input_data.get("joke", "")) > 1000,
            ],
            action=BlockAction(),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Length Limiter",
        ),
        *UiPathDeterministicGuardrailMiddleware(
            tools=[analyze_joke_syntax],
            rules=[],
            action=CustomFilterAction(
                word_to_filter="words",
                replacement="words++",
            ),
            stage=GuardrailExecutionStage.POST,
            name="Joke Content Always Filter",
        ),
        # LLM-as-judge: family-appropriateness filter. PRE_AND_POST so it judges
        # both the requested topic (input) and the produced joke (output). Block
        # action aborts the run on a violation — the input topic is rejected at
        # AGENT PRE before the LLM is ever called.
        *UiPathLLMAsJudgeMiddleware(
            name="Family Friendly Topic Judge",
            scopes=[GuardrailScope.AGENT],
            stage=GuardrailExecutionStage.PRE_AND_POST,
            action=BlockAction(),
            guardrail_text=(
                "The content must be appropriate for children and families. Flag it "
                "if the requested topic or the joke involves drugs, alcohol, sexual "
                "content, violence, weapons, hate, or anything unsuitable for kids."
            ),
            model="gpt-4o-2024-08-06",
            positive_examples=[
                "Generate a family-friendly joke based on the topic: banana"
            ],
            negative_examples=[
                "Generate a family-friendly joke based on the topic: buying cocaine"
            ],
            threshold=2,
            enabled_for_evals=False,
        ),
    ],
)


# Wrapper node to convert topic input to messages and call the agent. The
# guardrail middleware runs inside the agent; when the PII escalation guardrail
# fires, interrupt(CreateEscalation(...)) suspends the run for human review.
async def joke_node(state: Input) -> Output:
    """Convert topic to messages, call agent, and extract joke."""
    messages = [
        HumanMessage(
            content=f"Generate a family-friendly joke based on the topic: {state.topic}"
        )
    ]
    result = await agent.ainvoke({"messages": messages})
    joke = result["messages"][-1].content
    return Output(joke=joke)


# Build wrapper graph with custom input/output schemas. The runtime recompiles
# this with a durable checkpointer, so interrupt()/resume works under `uipath run`.
builder = StateGraph(Input, input_schema=Input, output_schema=Output)
builder.add_node("joke", joke_node)
builder.add_edge(START, "joke")
builder.add_edge("joke", END)

# Compile the graph
graph = builder.compile()
