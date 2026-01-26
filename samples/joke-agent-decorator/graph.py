"""Joke generating agent that creates family-friendly jokes based on a topic."""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel

from uipath.agent.models.agent import AgentGuardrailSeverityLevel
from uipath_langchain.chat import UiPathChat
from uipath_langchain.guardrails import (
    Entity,
    LogAction,
    PIIDetectionEntity,
    pii_guardrail,
)


# Define input schema for the agent
class Input(BaseModel):
    """Input schema for the joke agent."""
    topic: str


class Output(BaseModel):
    """Output schema for the joke agent."""
    joke: str


# Initialize UiPathChat LLM with PII guardrail decorator
# Option 1: Factory function (current approach - uses decorator syntax)
@pii_guardrail(
    entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
    action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
    name="LLM PII Detection",
)
def create_llm():
    """Create LLM instance with guardrails."""
    return UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)


llm = create_llm()

# Option 2: Direct function call (alternative - no decorator syntax)
# llm = pii_guardrail(
#     entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
#     action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
#     name="LLM PII Detection",
# )(UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7))

# Option 3: Helper function (if you want to reuse guardrail config)
# def create_guarded_llm(model: str, temperature: float = 0.7):
#     """Create LLM with standard PII guardrail."""
#     return pii_guardrail(
#         entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
#         action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
#         name="LLM PII Detection",
#     )(UiPathChat(model=model, temperature=temperature))
# llm = create_guarded_llm("gpt-4o-2024-08-06")


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

# Create agent with PII guardrail decorator
@pii_guardrail(
    entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
    action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
    name="Agent PII Detection",
)
def create_joke_agent():
    """Create the joke agent with guardrails."""
    return create_agent(
        model=llm,
        tools=[analyze_joke_syntax],
        system_prompt=SYSTEM_PROMPT,
    )


agent = create_joke_agent()


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
