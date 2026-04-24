from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from uipath_langchain.chat import (
    UiPathAzureChatOpenAI,
    UiPathChatAnthropicBedrock,
    UiPathChatGoogleGenerativeAI,
)

# Choose your LLM provider by uncommenting one of the following:
llm = UiPathChatAnthropicBedrock(model="anthropic.claude-haiku-4-5-20251001-v1:0")
# llm = UiPathAzureChatOpenAI(model="gpt-4.1-mini-2025-04-14")
# llm = UiPathChatGoogleGenerativeAI(model="gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's query using the available tools when needed. "
    "Be concise and informative."
)

MAX_RESPONSE_LENGTH = 5000


class InputModel(BaseModel):
    question: str = Field(
        description="Question for the assistant, e.g. 'What's the weather in Paris?'"
    )


class AgentResponse(TypedDict):
    response: str


class StateModel(BaseModel):
    messages: Annotated[list[Any], add_messages] = []
    structured_response: AgentResponse | None = None


@tool
def get_current_time() -> str:
    """Get the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


@tool
def get_weather(city: str, utc_time: str) -> str:
    """Get the current weather for a city. Requires the current UTC time from get_current_time.

    Args:
        city: The city name, e.g. 'Paris' or 'Tokyo'.
        utc_time: The current UTC time.
    """

    WEATHER_DATA = {
        "paris": "Weather in Paris, France: 18°C, wind 12 km/h, partly cloudy",
        "london": "Weather in London, UK: 14°C, wind 20 km/h, overcast",
        "new york": "Weather in New York, USA: 22°C, wind 8 km/h, clear sky",
        "tokyo": "Weather in Tokyo, Japan: 26°C, wind 5 km/h, sunny",
        "sydney": "Weather in Sydney, Australia: 19°C, wind 15 km/h, light rain",
    }

    weather = WEATHER_DATA.get(city.lower().strip())
    if weather:
        return f"{weather} (as of {utc_time})"
    return f"Weather data not available for {city}"


async def prepare(input: InputModel) -> dict[str, Any]:
    return {
        "messages": [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(input.question),
        ],
    }


react_agent = create_agent(
    model=llm,
    tools=[get_current_time, get_weather],
    response_format=AgentResponse,
)


async def postprocess(state: StateModel) -> AgentResponse:
    assert state.structured_response is not None, "Agent did not produce a structured response"
    response = state.structured_response["response"]
    if len(response) > MAX_RESPONSE_LENGTH:
        return AgentResponse(response=response[:MAX_RESPONSE_LENGTH] + "...")
    return state.structured_response


builder = StateGraph(StateModel, input=InputModel, output=AgentResponse)

builder.add_node("prepare", prepare)
builder.add_node("react_agent", react_agent)
builder.add_node("postprocess", postprocess)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "react_agent")
builder.add_edge("react_agent", "postprocess")
builder.add_edge("postprocess", END)

graph = builder.compile()
