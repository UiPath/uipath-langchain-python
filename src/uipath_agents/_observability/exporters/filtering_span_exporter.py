"""Filtering span exporter for OpenTelemetry."""

from typing import Callable, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class FilteringSpanExporter(SpanExporter):
    """Wraps a SpanExporter to filter spans before export."""

    def __init__(
        self,
        delegate: SpanExporter,
        filter_fn: Callable[[ReadableSpan], bool],
    ):
        """Initialize the filtering exporter.

        Args:
            delegate: The underlying exporter to send filtered spans to.
            filter_fn: Function that returns True for spans to export.
        """
        self._delegate = delegate
        self._filter_fn = filter_fn

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export only spans that pass the filter."""
        filtered = [s for s in spans if self._filter_fn(s)]
        if not filtered:
            return SpanExportResult.SUCCESS
        return self._delegate.export(filtered)

    def upsert_span(
        self,
        span: ReadableSpan,
        status_override: Optional[int] = None,
    ) -> SpanExportResult:
        """Upsert a single span, applying filter first."""
        if not self._filter_fn(span):
            return SpanExportResult.SUCCESS
        return self._delegate.upsert_span(span, status_override)  # type: ignore[attr-defined]

    def shutdown(self) -> None:
        """Shutdown the delegate exporter."""
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the delegate exporter."""
        return self._delegate.force_flush(timeout_millis)
