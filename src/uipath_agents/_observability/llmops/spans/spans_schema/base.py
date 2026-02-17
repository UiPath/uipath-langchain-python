"""Base utilities for span schemas.

Shared components used across all span schema modules.
"""

import json
from contextvars import ContextVar
from typing import Any, Dict, Optional, Protocol

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import (
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    Tracer,
)
from uipath.tracing import SpanStatus

from uipath_agents._errors import ExceptionMapper

from ..span_attributes import (
    BaseSpanAttributes,
    get_agent_version,
    get_execution_type,
)

__all__ = [
    "SyntheticReadableSpan",
    "SpanUpsertProtocol",
    "reference_id_context",
    "uipath_source_context",
    "apply_attributes",
    "get_parent_context",
    "create_span",
    "end_span_ok",
    "end_span_error",
    "format_span_error",
]

# Context variable to propagate reference_id to all spans in a trace
reference_id_context: ContextVar[Optional[str]] = ContextVar(
    "reference_id", default=None
)

uipath_source_context: ContextVar[Optional[int]] = ContextVar(
    "uipath_source", default=None
)


class SyntheticReadableSpan:
    """Minimal ReadableSpan for upsert from saved data after process restart."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: Dict[str, Any],
        parent_span_id: Optional[str] = None,
    ):
        trace_id_int = int(trace_id, 16)
        span_id_int = int(span_id, 16)
        parent_id_int = int(parent_span_id, 16) if parent_span_id else None

        self.name = name
        self.start_time = start_time_ns
        self.end_time = end_time_ns
        self.attributes = attributes
        self.status = Status(StatusCode.OK)
        self.kind = SpanKind.INTERNAL
        self.events: tuple[Any, ...] = ()
        self.links: tuple[Any, ...] = ()
        self.resource = None
        self.instrumentation_info = None
        self.parent = (
            SpanContext(
                trace_id=trace_id_int,
                span_id=parent_id_int,
                is_remote=False,
                trace_flags=TraceFlags(0x01),
            )
            if parent_id_int
            else None
        )
        self._span_context = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )

    def get_span_context(self) -> SpanContext:
        return self._span_context


class SpanUpsertProtocol(Protocol):
    """Protocol for span upsert operations."""

    def upsert_span(
        self,
        span: ReadableSpan,
        status_override: Optional[int] = None,
    ) -> SpanExportResult: ...


def apply_attributes(span: Span, attrs: BaseSpanAttributes) -> None:
    """Apply typed attributes to an OpenTelemetry span.

    OTEL only accepts primitives. Complex objects (dict/list) are JSON
    serialized here. To avoid double-escaping in storage, the span
    processor should handle final serialization.

    Automatically populates execution_type, agent_version, and reference_id
    on all spans for consistent telemetry.

    Args:
        span: The span to set attributes on
        attrs: Typed attributes to apply
    """
    for key, value in attrs.to_otel_attributes().items():
        if value is None:
            continue
        # OTEL only accepts primitives - serialize complex objects
        if isinstance(value, (dict, list)):
            span.set_attribute(key, json.dumps(value))
        else:
            span.set_attribute(key, value)

    # Execution context fields on ALL spans for consistent telemetry
    span.set_attribute("executionType", get_execution_type())
    agent_version = get_agent_version()
    if agent_version:
        span.set_attribute("agentVersion", agent_version)
    ref_id = reference_id_context.get()
    if ref_id:
        span.set_attribute("referenceId", ref_id)
    uipath_src = uipath_source_context.get()
    if uipath_src is not None:
        span.set_attribute("uipath.source", uipath_src)


def get_parent_context(
    parent_span: Optional[Span] = None,
) -> Optional[Any]:
    """Get parent context for span creation.

    Args:
        parent_span: Optional parent span object. If None, uses current span.

    Returns:
        OpenTelemetry context for the parent span
    """
    parent = parent_span or trace.get_current_span()
    return trace.set_span_in_context(parent) if parent else None


def create_span(
    tracer: Tracer,
    name: str,
    parent_span: Optional[Span] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Span:
    """Create a new span with proper parent context.

    Args:
        tracer: The OpenTelemetry tracer to use
        name: Name of the span
        parent_span: Optional parent span object. If None, uses current span.
        kind: Span kind (default INTERNAL)

    Returns:
        The created Span
    """
    context = get_parent_context(
        parent_span=parent_span,
    )
    return tracer.start_span(name, kind=kind, context=context)


def end_span_ok(
    span: Span,
    upsert_fn: Optional[Any] = None,
) -> None:
    """End a span with OK status.

    Args:
        span: The span to end
        upsert_fn: Optional function to call for upsert
    """
    span.set_status(Status(StatusCode.OK))
    span.end()
    if upsert_fn:
        upsert_fn(span, status=SpanStatus.OK)


def format_span_error(error: Exception) -> str:
    """Format an exception into a span error message via ExceptionMapper."""
    error_info = ExceptionMapper.map_runtime(error).error_info
    return (
        f"{error_info.title}\nDetails:\n{error_info.detail}\nCode: {error_info.code}\n"
    )


def end_span_error(
    span: Span,
    error: Exception,
    upsert_fn: Optional[Any] = None,
) -> None:
    """End a span with ERROR status.

    Args:
        span: The span to end
        error: The exception that caused the error
        upsert_fn: Optional function to call for upsert
    """
    # May be overridden by _SpanUtils.otel_span_to_uipath_span during export
    span.set_attribute("error", format_span_error(error))
    span.set_status(Status(StatusCode.ERROR, format_span_error(error)))
    span.end()
    if upsert_fn:
        upsert_fn(span, status=SpanStatus.ERROR)
