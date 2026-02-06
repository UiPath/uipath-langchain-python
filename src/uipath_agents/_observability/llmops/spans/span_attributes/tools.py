"""Tool-specific span attribute classes."""

from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from .base import BaseSpanAttributes
from .tool_call import ToolCallSpanAttributes
from .types import SpanType


class ProcessToolSpanAttributes(BaseSpanAttributes):
    """Attributes for UiPath process tool calls."""

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(..., alias="toolName")
    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    result: Optional[Any] = Field(None, alias="result")
    job_id: Optional[str] = Field(None, alias="jobId")
    job_details_uri: Optional[str] = Field(None, alias="jobDetailsUri")

    @property
    def type(self) -> str:
        return SpanType.PROCESS_TOOL


class ActionCenterToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for Action Center tool calls."""

    @property
    def type(self) -> str:
        return SpanType.ACTION_CENTER_TOOL


class AgentToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for agent-as-tool spans."""

    job_id: Optional[str] = Field(None, alias="jobId")
    job_details_uri: Optional[str] = Field(None, alias="jobDetailsUri")

    @property
    def type(self) -> str:
        return SpanType.AGENT_TOOL


class ApiWorkflowToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for API workflow tool spans."""

    job_id: Optional[str] = Field(None, alias="jobId")
    job_details_uri: Optional[str] = Field(None, alias="jobDetailsUri")

    @property
    def type(self) -> str:
        return SpanType.API_WORKFLOW_TOOL


class AgenticProcessToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for agentic process tool spans."""

    job_id: Optional[str] = Field(None, alias="jobId")
    job_details_uri: Optional[str] = Field(None, alias="jobDetailsUri")

    @property
    def type(self) -> str:
        return SpanType.AGENTIC_PROCESS_TOOL


class EscalationToolSpanAttributes(BaseSpanAttributes):
    """Attributes for escalation tool spans.

    Child span under tool call containing escalation details.
    """

    model_config = ConfigDict(populate_by_name=True)

    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    channel_type: Optional[str] = Field(None, alias="channelType")
    assigned_to: Optional[str] = Field(None, alias="assignedTo")
    task_id: Optional[str] = Field(None, alias="taskId")
    task_url: Optional[str] = Field(None, alias="taskUrl")
    result: Optional[Any] = Field(None, alias="result")
    from_memory: Optional[bool] = Field(None, alias="fromMemory")
    saved_to_memory: Optional[bool] = Field(None, alias="savedToMemory")

    @property
    def type(self) -> str:
        return SpanType.ESCALATION_TOOL


class IntegrationToolSpanAttributes(BaseSpanAttributes):
    """Attributes for integration tool spans.

    Child span under Tool call for integration tool execution.
    Replaces the SDK's activity_invoke span which gets filtered.
    """

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(..., alias="toolName")
    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    result: Optional[Any] = Field(None, alias="result")

    @property
    def type(self) -> str:
        return SpanType.INTEGRATION_TOOL


class InternalToolSpanAttributes(BaseSpanAttributes):
    """Attributes for internal tool spans."""

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(..., alias="toolName")
    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")

    @property
    def type(self) -> str:
        return SpanType.TOOL_CALL
