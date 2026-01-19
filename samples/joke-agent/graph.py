"""Joke generating agent that creates family-friendly jokes based on a topic."""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel
from uipath.core.guardrails import GuardrailScope

from middleware import CustomFilterAction, LoggingMiddleware
from uipath_langchain.chat import UiPathChat
from uipath_langchain.guardrails import (
    BlockAction,
    PIIDetectionEntity,
    GuardrailExecutionStage,
    LogAction,
    PIIDetectionEntityType,
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
)
from uipath_langchain.guardrails.actions import LoggingSeverityLevel


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
        *UiPathPIIDetectionMiddleware(
            name="My personal PII detector",
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[
                PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.CREDIT_CARD_NUMBER, 0.5),
            ],
        ),
        *UiPathPIIDetectionMiddleware(
            name="Tool PII detector",
            scopes=[GuardrailScope.TOOL],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[
                PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.CREDIT_CARD_NUMBER, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.PHONE_NUMBER, 0.5),
            ],
            tools=[analyze_joke_syntax],
        ),
        *UiPathPromptInjectionMiddleware(
            name="Prompt Injection Detection",
            scopes=[GuardrailScope.LLM],
            action=BlockAction(),
            threshold=0.5,
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
        )
    ],
)


# Wrapper node to convert topic input to messages and call the agent
async def joke_node(state: Input) -> Output:
    """Convert topic to messages, call agent, and extract joke."""
    # Convert topic to messages format
    messages = [
        HumanMessage(content=f"Generate a family-friendly joke based on the topic: {state.topic}")
    ]

    # Call the agent with messages
    result = await agent.ainvoke({"messages": messages})

    # Extract the joke from the agent's response
    joke = result["messages"][-1].content

    return Output(joke=joke)


# Build wrapper graph with custom input/output schemas
builder = StateGraph(Input, input=Input, output=Output)
builder.add_node("joke", joke_node)
builder.add_edge(START, "joke")
builder.add_edge("joke", END)

# Compile the graph
graph = builder.compile()
