from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from uipath_langchain.chat import UiPathChatAnthropicBedrock

# Goes through the LLM Gateway, which traces it, so it has no client-side LLM span.
llm = UiPathChatAnthropicBedrock(model="anthropic.claude-haiku-4-5-20251001-v1:0")

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's query using the available tools when needed. "
    "Be concise and informative."
)


class FakeDirectProviderChatModel(BaseChatModel):
    """Stands in for a direct provider model: it keeps its client-side LLM span."""

    model_name: str = "fake-direct-provider-1"

    @property
    def _llm_type(self) -> str:
        return "fake-direct-provider"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = str(messages[-1].content) if messages else ""
        reply = f"[fake direct provider] Received {len(last)} characters: {last[:120]}"
        message = AIMessage(
            content=reply,
            usage_metadata={
                "input_tokens": len(last.split()),
                "output_tokens": len(reply.split()),
                "total_tokens": len(last.split()) + len(reply.split()),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


direct_llm = FakeDirectProviderChatModel()


class InputModel(BaseModel):
    question: str = Field(description="Question for the assistant")


class AgentResponse(TypedDict):
    response: str


class OutputModel(TypedDict):
    response: str
    direct_model_reply: str


class StateModel(BaseModel):
    messages: Annotated[list[Any], add_messages] = []
    structured_response: AgentResponse | None = None
    direct_model_reply: str = ""


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
    weather_data = {
        "paris": "Weather in Paris, France: 18°C, wind 12 km/h, partly cloudy",
        "london": "Weather in London, UK: 14°C, wind 20 km/h, overcast",
        "tokyo": "Weather in Tokyo, Japan: 26°C, wind 5 km/h, sunny",
    }
    weather = weather_data.get(city.lower().strip())
    if weather:
        return f"{weather} (as of {utc_time})"
    return f"Weather data not available for {city}"


async def prepare(input: InputModel) -> dict[str, Any]:
    return {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(input.question)]}


react_agent = create_agent(
    model=llm,
    tools=[get_current_time, get_weather],
    response_format=AgentResponse,
)


async def direct_model(state: StateModel) -> dict[str, Any]:
    assert state.structured_response is not None, "Agent did not produce a structured response"
    reply = await direct_llm.ainvoke([HumanMessage(state.structured_response["response"])])
    return {"direct_model_reply": str(reply.content)}


async def postprocess(state: StateModel) -> OutputModel:
    assert state.structured_response is not None, "Agent did not produce a structured response"
    return OutputModel(
        response=state.structured_response["response"],
        direct_model_reply=state.direct_model_reply,
    )


builder = StateGraph(StateModel, input=InputModel, output=OutputModel)
builder.add_node("prepare", prepare)
builder.add_node("react_agent", react_agent)
builder.add_node("direct_model", direct_model)
builder.add_node("postprocess", postprocess)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "react_agent")
builder.add_edge("react_agent", "direct_model")
builder.add_edge("direct_model", "postprocess")
builder.add_edge("postprocess", END)

graph = builder.compile()
