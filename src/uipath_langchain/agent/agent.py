from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from uipath_langchain.agent.state import LowCodeAgentState


def create_lowcode_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | ToolNode,
    messages: Sequence[SystemMessage | HumanMessage],
    *,
    response_format: type[BaseModel] | None = None,
) -> StateGraph[LowCodeAgentState]:
    """Create a LangGraph agent with the LowCode agent pattern.

    Args:
        model: The language model to use for the agent
        tools: Sequence of tools available to the agent
        messages: Initial messages to populate the graph state
        response_format: Optional structured output schema for agent

    Returns:
        StateGraph configured with model and tools executing the LowCode agent
    """
    builder = StateGraph(LowCodeAgentState)
    builder.add_edge(START, END)

    return builder
