from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from uipath.tracing import traced

from uipath_langchain.chat import GeminiModels, OpenAIModels, UiPathChatOpenAI
from uipath_langchain.chat.vertex import UiPathChatVertex
SYSTEM_PROMPT = ( 
    "You are a helpful assistant. "
    "Answer the user's query using the available tools when needed."
    "Be concise and informative."
)


class InputModel(BaseModel):
    query: str
    refine: bool = False


class AgentResponse(TypedDict):
    response: str


class RefinementSuggestion(BaseModel):
    suggestion: str = Field(description="A specific suggestion to improve the agent's response")


class StateModel(BaseModel):
    messages: Annotated[list, add_messages] = []
    structured_response: AgentResponse | None = None
    refine: bool = False
    was_refined: bool = False
    result: str = ""


class OutputModel(BaseModel):
    result: str


@tool
@traced
def get_current_time() -> str:
    """Get the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


web_search = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for information.",
    api_wrapper=DuckDuckGoSearchAPIWrapper(backend="duckduckgo"),
)


async def prepare(input: InputModel) -> dict:
    return {
        "messages": [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(input.query),
        ],
        "refine": input.refine,
    }


react_agent = create_agent(
    model=UiPathChatOpenAI(model_name=OpenAIModels.gpt_4_1_mini_2025_04_14),
    tools=[get_current_time, web_search],
    response_format=AgentResponse,
)


async def refine(state: StateModel) -> dict:
    """Optionally refine the agent response using a quality reviewer."""
    if state.refine and not state.was_refined:
        suggestion = await UiPathChatVertex(model_name=GeminiModels.gemini_2_5_flash).with_structured_output(RefinementSuggestion).ainvoke(
            [
                SystemMessage(
                    "You are a quality reviewer. Based on the topic and the current response, "
                    "suggest one specific improvement to make the answer more accurate or complete."
                ),
                HumanMessage(
                    f"Topic: {state.messages[1].content}\n"
                    f"Current response: {state.structured_response['response']}"
                ),
            ]
        )
        return {
            "messages": [HumanMessage(f"Refinement suggestion: {suggestion.suggestion}")],
            "was_refined": True,
        }
    return {"result": state.structured_response["response"]}


def route(state: StateModel) -> Literal["react_agent", END]:
    if state.refine and state.was_refined and not state.result:
        return "react_agent"
    return END


builder = StateGraph(StateModel, input=InputModel, output=OutputModel)

builder.add_node("prepare", prepare)
builder.add_node("react_agent", react_agent)
builder.add_node("refine", refine)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "react_agent")
builder.add_edge("react_agent", "refine")
builder.add_conditional_edges("refine", route)

graph = builder.compile()
