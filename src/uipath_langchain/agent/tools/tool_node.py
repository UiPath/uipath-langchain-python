"""Tool node factory wiring directly to LangGraph's ToolNode."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from uipath.agent.models.agent import AgentGuardrail, GuardrailAction

from uipath_langchain.agent.react.guardrails_subgraph import create_tool_guardrails_subgraph


def create_tool_node(tools: Sequence[BaseTool], guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None) -> \
    dict[str, ToolNode]:
    """Create individual ToolNode for each tool.

    Args:
        tools: Sequence of tools to create nodes for.

    Returns:
        Dict mapping tool.name -> ToolNode([tool]).
        Each tool gets its own dedicated node for middleware composition.

    Note:
        handle_tool_errors=False delegates error handling to LangGraph's error boundary.
    """
    result: dict[str, ToolNode] = {}
    for tool in tools:
        subgraph = create_tool_guardrails_subgraph((tool.name + "Exec", ToolNode([tool], handle_tool_errors=False)),
                                                   guardrails)
        result[tool.name] = subgraph

    return result
