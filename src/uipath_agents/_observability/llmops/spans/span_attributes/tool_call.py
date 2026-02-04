"""Core tool span attribute classes."""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field
from uipath.tracing import SpanAttachment

from .base import BaseSpanAttributes
from .types import SpanType


class ToolCallSpanAttributes(BaseSpanAttributes):
    """Attributes for tool call spans."""

    model_config = ConfigDict(populate_by_name=True)

    call_id: Optional[str] = Field(None, alias="callId")
    tool_name: str = Field(..., alias="toolName")
    tool_type: str = Field(default="toolCall", alias="toolType")
    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    result: Optional[Any] = Field(None, alias="result")
    attachments: Optional[List[SpanAttachment]] = Field(None, alias="attachments")
    _span_type: Optional[str] = None

    def __init__(self, span_type: Optional[str] = None, **data: Any):
        super().__init__(**data)
        self._span_type = span_type

    @property
    def type(self) -> str:
        return self._span_type or self.tool_type


class ToolExecutionSpanAttributes(BaseSpanAttributes):
    """Attributes for tool execution spans (inner execution)."""

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(..., alias="toolName")

    @property
    def type(self) -> str:
        return SpanType.TOOL_EXECUTION
