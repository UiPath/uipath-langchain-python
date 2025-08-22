import json
import logging
import os
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from uipath.tracing._utils import _SpanUtils

from .LangchainSpanProcessor import BaseSpanProcessor

logger = logging.getLogger(__name__)


class JsonFileExporter(SpanExporter):
    """
    An exporter that writes spans to a file in JSON Lines format.

    This exporter is useful for debugging and local development. It serializes
    each span to a JSON object and appends it as a new line in the specified
    file.
    """

    def __init__(self, file_path: str, processor: BaseSpanProcessor):
        """
        Initializes the JsonFileExporter.

        Args:
            file_path: The path to the JSON file where spans will be written.
        """
        self.file_path = file_path
        self._processor = processor
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Exports a batch of spans.

        Args:
            spans: A sequence of ReadableSpan objects.

        Returns:
            The result of the export operation.
        """
        try:
            uipath_spans = [
                _SpanUtils.otel_span_to_uipath_span(span).to_dict() for span in spans
            ]
            processed_spans = [
                self._processor.process_span(span) for span in uipath_spans
            ]
            with open(self.file_path, "a") as f:
                for span in processed_spans:
                    f.write(json.dumps(span) + "\n")
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to export spans to {self.file_path}: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shuts down the exporter."""
        pass
