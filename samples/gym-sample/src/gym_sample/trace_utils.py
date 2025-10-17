"""OpenTelemetry trace collection utilities for agent evaluation."""

from collections.abc import Sequence
from typing import List

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from openinference.instrumentation.langchain import LangChainInstrumentor


class InMemorySpanExporter(SpanExporter):
    """An OpenTelemetry span exporter that stores spans in memory for testing."""

    def __init__(self):
        self.spans = []
        self.is_shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self.is_shutdown:
            return SpanExportResult.FAILURE

        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_exported_spans(self) -> List[ReadableSpan]:
        return self.spans

    def clear_exported_spans(self) -> None:
        self.spans = []

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return not self.is_shutdown

    def shutdown(self) -> None:
        self.is_shutdown = True


def setup_tracer() -> tuple[InMemorySpanExporter, TracerProvider]:
    """Set up OpenTelemetry tracing with LangChain instrumentation.

    Returns:
        InMemorySpanExporter: The configured span exporter for capturing traces.
        TracerProvider: The configured tracer provider for capturing traces.
    """
    # Set up OpenTelemetry trace collection
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Initialize LangChain instrumentation (this creates the spans!)
    LangChainInstrumentor().instrument()

    return exporter, provider
