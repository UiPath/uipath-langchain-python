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

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

from .analyze_files_tool import create_analyze_file_tool
from .batch_transform_tool import create_batch_transform_tool
from .deeprag_tool import create_deeprag_tool

_INTERNAL_TOOL_HANDLERS: dict[
    AgentInternalToolType,
    Callable[[AgentInternalToolResourceConfig, BaseChatModel], StructuredTool],
] = {
    AgentInternalToolType.ANALYZE_FILES: create_analyze_file_tool,
    AgentInternalToolType.DEEP_RAG: create_deeprag_tool,
    AgentInternalToolType.BATCH_TRANSFORM: create_batch_transform_tool,
}


def create_internal_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create an internal tool based on the resource configuration.

    Raises:
        AgentStartupError: If the tool type is not supported (no handler exists for it).

    """
    tool_type = resource.properties.tool_type

    handler = _INTERNAL_TOOL_HANDLERS.get(tool_type)
    if handler is None:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Unsupported internal tool type",
            detail=f"Unsupported internal tool type: {tool_type}. "
            f"Supported types: {list[AgentInternalToolType](_INTERNAL_TOOL_HANDLERS.keys())}.",
            category=UiPathErrorCategory.USER,
        )

    return handler(resource, llm)
