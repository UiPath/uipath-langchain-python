"""MCP-related span attribute classes."""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from .base import BaseSpanAttributes
from .types import SpanType


class McpToolSpanAttributes(BaseSpanAttributes):
    """Attributes for MCP tool spans."""

    model_config = ConfigDict(populate_by_name=True)

    arguments: Dict[str, Any] = Field(..., alias="arguments")
    result: Optional[Any] = Field(None, alias="result")

    @property
    def type(self) -> str:
        return SpanType.MCP_TOOL


class McpSessionStartSpanAttributes(BaseSpanAttributes):
    """Attributes for MCP session start spans."""

    model_config = ConfigDict(populate_by_name=True)

    mcp_servers: List[str] = Field(..., alias="mcpServers")

    @property
    def type(self) -> str:
        return SpanType.MCP_SESSION_START


class McpSessionStopSpanAttributes(BaseSpanAttributes):
    """Attributes for MCP session stop spans."""

    model_config = ConfigDict(populate_by_name=True)

    mcp_servers: List[str] = Field(..., alias="mcpServers")

    @property
    def type(self) -> str:
        return SpanType.MCP_SESSION_STOP
