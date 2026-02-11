"""Tool span schemas.

Handles tool call, escalation, process, agent, and integration tool spans.
"""

from typing import Any, Callable, Dict, Optional, Type

from opentelemetry.trace import (
    Span,
    SpanKind,
    Tracer,
)
from pydantic import BaseModel
from uipath.tracing import AttachmentDirection

from ...instrumentors.attribute_helpers import get_span_attachments
from ..span_attributes import (
    AgentToolSpanAttributes,
    EscalationToolSpanAttributes,
    IntegrationToolSpanAttributes,
    InternalToolSpanAttributes,
    ProcessToolSpanAttributes,
    SpanType,
    ToolCallSpanAttributes,
)
from ..span_name import SpanName
from .base import apply_attributes, create_span

__all__ = [
    "ToolSpanSchema",
]


class ToolSpanSchema:
    """Schema for tool-related spans."""

    def __init__(
        self,
        tracer: Tracer,
        upsert_started_fn: Optional[Callable[[Span], bool]] = None,
    ):
        """Initialize tool span schema.

        Args:
            tracer: The OpenTelemetry tracer to use
            upsert_started_fn: Optional function to upsert span on start
        """
        self._tracer = tracer
        self._upsert_started = upsert_started_fn

    def start_tool_call(
        self,
        tool_name: str,
        tool_type: str = SpanType.TOOL_CALL,
        tool_type_value: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        parent_span: Optional[Span] = None,
        args_schema: Optional[Type[BaseModel]] = None,
    ) -> Span:
        """Start a tool call span.

        Args:
            tool_name: Name of the tool being called
            tool_type: Span type string (toolCall, processTool, etc.)
            tool_type_value: Tool type for display (Agent, Process, Integration)
            arguments: Arguments passed to the tool
            call_id: LLM tool call ID
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        span = create_span(
            self._tracer,
            SpanName.tool_call(tool_name),
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )

        attachments = get_span_attachments(
            arguments, args_schema, direction=AttachmentDirection.IN
        )
        attrs = ToolCallSpanAttributes(
            tool_name=tool_name,
            span_type=tool_type,
            tool_type=tool_type_value or "Integration",
            arguments=arguments,
            call_id=call_id,
            attachments=attachments,
        )
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

    def start_escalation_tool(
        self,
        app_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        channel_type: Optional[str] = None,
        assignee: Optional[str] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an escalation tool span (child of tool call).

        Creates a span named after the action app for human-in-the-loop escalations.
        The span name reflects the escalation destination (e.g., "SimpleApprovalApp").

        Args:
            app_name: Name of the action center app (used as span name)
            arguments: Arguments passed to the escalation
            channel_type: Type of channel (e.g., "actionCenter")
            assignee: Who the task is assigned to
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        span = create_span(
            self._tracer,
            app_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )
        attrs = EscalationToolSpanAttributes(
            arguments=arguments,
            channel_type=channel_type,
            assigned_to=assignee,
        )
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

    def start_process_tool(
        self,
        process_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a process tool span (child of tool call).

        Creates a span named after the process for interruptible process calls.

        Args:
            process_name: Display name of the UiPath process (used as span name)
            arguments: Arguments passed to the process
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        span = create_span(
            self._tracer,
            process_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )
        attrs = ProcessToolSpanAttributes(
            arguments=arguments,
        )
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

    def start_agent_tool(
        self,
        agent_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an agent tool span (child of tool call for agent-as-tool).

        Creates a span named after the invoked agent.
        The span name reflects which agent was called (e.g., "A_plus_B").

        Args:
            agent_name: Name of the agent (used as span name)
            arguments: Arguments passed to the agent
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        span = create_span(
            self._tracer,
            agent_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )
        attrs = AgentToolSpanAttributes(
            tool_name=agent_name,
            arguments=arguments,
        )
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

    def start_integration_tool(
        self,
        tool_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an integration tool span (child of tool call).

        Creates a child span for integration tool execution. This replaces
        the SDK's activity_invoke span which gets filtered out from LLMOps.

        Args:
            tool_name: Name of the integration tool (used as span name)
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        span = create_span(
            self._tracer,
            tool_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )
        attrs = IntegrationToolSpanAttributes(tool_name=tool_name, arguments=arguments)
        apply_attributes(span, attrs)
        # Note: integration tool doesn't upsert on start (short-lived span)
        return span

    def start_internal_tool(
        self,
        tool_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        span = create_span(
            self._tracer,
            tool_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )
        attrs = InternalToolSpanAttributes(tool_name=tool_name, arguments=arguments)
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span
