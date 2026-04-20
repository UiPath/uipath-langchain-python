from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from uipath_langchain.chat import (
    BedrockModels,
    GeminiModels,
    OpenAIModels,
    UiPathChatBedrock,
    UiPathChatOpenAI,
    UiPathChatVertex,
)

# Choose your LLM provider by uncommenting one of the following:
llm = UiPathChatVertex(model_name=GeminiModels.gemini_2_5_flash)
# llm = UiPathChatBedrock(model_name=BedrockModels.anthropic_claude_haiku_4_5)
# llm = UiPathChatOpenAI(model_name=OpenAIModels.gpt_4_1_mini_2025_04_14)

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's query using the available tools when needed. "
    "Be concise and informative."
)

MAX_RESPONSE_LENGTH = 5000


class InputModel(BaseModel):
    query: str


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
            HumanMessage(input.query),
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
