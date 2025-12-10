"""Tool node factory wiring directly to LangGraph's ToolNode."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from uipath_langchain._tracing.tracer import (
    get_tracer,
    is_custom_instrumentation_enabled,
)


def create_tool_node(tools: Sequence[BaseTool]) -> dict[str, Callable[..., Any]]:
    """Create individual ToolNode for each tool with optional tracing.

    Args:
        tools: Sequence of tools to create nodes for.

    Returns:
        Dict mapping tool.name -> traced tool node callable.
        Each tool gets its own dedicated node for middleware composition.

    Note:
        handle_tool_errors=False delegates error handling to LangGraph's error boundary.
        If UIPATH_CUSTOM_INSTRUMENTATION is enabled, wraps execution with tool call spans.
    """
    return {tool.name: _create_traced_tool_node(tool) for tool in tools}


def _create_traced_tool_node(tool: BaseTool) -> Callable[..., Any]:
    """Create a traced wrapper for a single tool."""
    base_node = ToolNode([tool], handle_tool_errors=False)

    async def traced_tool_node(state: dict[str, Any]) -> dict[str, Any]:
        if is_custom_instrumentation_enabled():
            tracer = get_tracer()
            with tracer.start_tool_call(tool.name):
                return await base_node.ainvoke(state)

        return await base_node.ainvoke(state)

    return traced_tool_node
