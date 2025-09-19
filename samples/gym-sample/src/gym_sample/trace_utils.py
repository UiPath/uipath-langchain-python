"""OpenTelemetry trace collection utilities for agent evaluation."""

from typing import List

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from openinference.instrumentation.langchain import LangChainInstrumentor


class SpanCollector(SpanExporter):
    """Span exporter that collects actual ReadableSpan objects."""

    def __init__(self) -> None:
        """Initialize the span collector."""
        self.spans: List[ReadableSpan] = []

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        """Export spans by collecting them."""
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush spans."""
        return True

    def get_spans(self) -> List[ReadableSpan]:
        """Get all collected spans."""
        return self.spans.copy()

    def clear_spans(self) -> None:
        """Clear all collected spans."""
        self.spans.clear()


def setup_tracing() -> SpanCollector:
    """Set up OpenTelemetry tracing with LangChain instrumentation.

    Returns:
        SpanCollector: The configured span collector for capturing traces.
    """
    # Create collector
    collector = SpanCollector()

    # Set up OpenTelemetry trace collection
    tracer_provider = TracerProvider()
    span_processor = SimpleSpanProcessor(collector)
    tracer_provider.add_span_processor(span_processor)

    # Set the tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Initialize LangChain instrumentation (this creates the spans!)
    LangChainInstrumentor().instrument()

    return collector
