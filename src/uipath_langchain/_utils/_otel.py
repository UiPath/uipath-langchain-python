"""OpenTelemetry utilities for span and trace context."""

import os
from typing import Any


def get_current_span_and_trace_ids() -> tuple[str, str]:
    """Get current OpenTelemetry span ID and trace ID.

    Returns hex-encoded IDs, or empty strings if no active span.
    Falls back to UIPATH_TRACE_ID env var for trace_id.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return (
                format(ctx.span_id, "016x"),
                format(ctx.trace_id, "032x"),
            )
    except ImportError:
        pass

    trace_id = os.environ.get("UIPATH_TRACE_ID", "")
    return ("", trace_id)


def set_span_attribute(name: str, value: Any) -> None:
    """Set an attribute on the current OTel span (no-op if unavailable)."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(name, value)
    except ImportError:
        pass
