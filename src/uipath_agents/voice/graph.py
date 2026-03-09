"""Voice agent graph and prompt utilities."""

from datetime import datetime, timezone

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import create_tool_node


def get_voice_system_prompt(
    system_message: str,
    agent_name: str | None,
) -> str:
    name = agent_name or "Voice Assistant"
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return f"You are {name}.\nThe current date is: {date}.\n\n{system_message}"


def build_voice_tool_graph(tool: BaseTool) -> StateGraph[AgentGraphState]:
    """Returns an uncompiled StateGraph — caller compiles with a checkpointer."""
    # handle_tool_errors=False so GraphInterrupt propagates for proper suspension
    tool_nodes = create_tool_node([tool], handle_tool_errors=False)
    node = tool_nodes[tool.name]

    graph = StateGraph(AgentGraphState)
    graph.add_node(tool.name, node)
    graph.add_edge(START, tool.name)
    graph.add_edge(tool.name, END)

    return graph
