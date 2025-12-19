"""Factory for creating internal agent tools.

This module provides a factory pattern for creating internal tools used by agents.
Internal tools are built-in tools that provide core functionality for agents, such as
file analysis, data processing, or other utilities that don't require external integrations.

Supported Internal Tools:
    - ANALYZE_FILES: Tool for analyzing file contents and extracting information

Example:
    >>> from uipath.agent.models.agent import AgentInternalToolResourceConfig
    >>> resource = AgentInternalToolResourceConfig(...)
    >>> tool = create_internal_tool(resource)
    >>> # Use the tool in your agent workflow
"""

from typing import Callable

from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
)

from .analyze_files_tool import create_analyze_file_tool

_INTERNAL_TOOL_HANDLERS: dict[
    AgentInternalToolType, Callable[[AgentInternalToolResourceConfig], StructuredTool]
] = {
    AgentInternalToolType.ANALYZE_FILES: create_analyze_file_tool,
}


def create_internal_tool(resource: AgentInternalToolResourceConfig) -> StructuredTool:
    """Create an internal tool based on the resource configuration.

    Args:
        resource: Internal tool resource configuration containing the tool type and
                 properties needed for tool creation.

    Returns:
        A LangChain StructuredTool instance configured for the specified internal tool.

    Raises:
        ValueError: If the tool type is not supported (no handler exists for it).

    Example:
        >>> resource = AgentInternalToolResourceConfig(
        ...     properties=AgentInternalToolProperties(
        ...         tool_type=AgentInternalToolType.ANALYZE_FILES
        ...     )
        ... )
        >>> tool = create_internal_tool(resource)
        >>> result = tool.invoke({"file_content": "..."})
    """
    tool_type = resource.properties.tool_type

    # Get the appropriate handler for this tool type
    handler = _INTERNAL_TOOL_HANDLERS.get(tool_type)
    if handler is None:
        raise ValueError(
            f"Unsupported internal tool type: {tool_type}. "
            f"Supported types: {list[AgentInternalToolType](_INTERNAL_TOOL_HANDLERS.keys())}"
        )

    return handler(resource)
