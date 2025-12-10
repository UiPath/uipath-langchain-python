"""Tool node factory wiring directly to LangGraph's ToolNode."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode


def create_tool_node(tools: Sequence[BaseTool]) -> dict[str, ToolNode]:
    """Create individual ToolNode for each tool.

    Args:
        tools: Sequence of tools to create nodes for.

    Returns:
        Dict mapping tool.name -> ToolNode([tool]).
        Each tool gets its own dedicated node for middleware composition.

    Note:
        handle_tool_errors=False delegates error handling to LangGraph's error boundary.
    """
    return {tool.name: ToolNode([tool], handle_tool_errors=False) for tool in tools}
