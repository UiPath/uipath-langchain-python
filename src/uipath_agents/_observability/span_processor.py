"""Span processor for marking spans to be filtered by exporters."""

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from .tracing import is_openinference_span


class SourceMarkerProcessor(SpanProcessor):
    """Marks spans for filtering by exporters.

    Tags OpenInference spans with telemetry.filter="drop".
    The LlmOpsHttpExporter skips spans marked for drop
    (they go to AppInsights instead of LLMOps).
    """

    FILTER_ATTRIBUTE = "telemetry.filter"

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Mark OpenInference spans to be dropped by LlmOpsHttpExporter."""
        if is_openinference_span(span):
            span.set_attribute(self.FILTER_ATTRIBUTE, "drop")

    def on_end(self, span: ReadableSpan) -> None:
        """No-op on span end."""
        pass

    def shutdown(self) -> None:
        """No-op shutdown."""
        pass

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        """No-op flush."""
        return True
