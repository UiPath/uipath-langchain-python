"""OpenTelemetry utilities for span and trace context."""

from typing import Any


def get_current_span_and_trace_ids() -> tuple[str, str]:
    """Return the current OTel span and trace IDs as hex strings."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        context = span.get_span_context()
        if not context.is_valid:
            return "", ""
        return f"{context.span_id:016x}", f"{context.trace_id:032x}"
    except ImportError:
        return "", ""


def set_span_attribute(name: str, value: Any) -> None:
    """Set an attribute on the current OTel span (no-op if unavailable)."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(name, value)
    except ImportError:
        pass


def set_current_span_error(error: BaseException) -> None:
    """Record an exception and mark the current OTel span as errored."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode

        span = trace.get_current_span()
        if span.is_recording():
            span.record_exception(error)
            span.set_status(StatusCode.ERROR, str(error))
    except ImportError:
        pass
