"""OpenTelemetry utilities for span and trace context."""

from typing import Any


def set_span_attribute(name: str, value: Any) -> None:
    """Set an attribute on the current OTel span (no-op if unavailable)."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(name, value)
    except ImportError:
        pass
