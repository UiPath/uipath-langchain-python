"""File exporter for UiPath internal trace format.

Exports spans in the same format as LlmOpsHttpExporter but writes to a JSON Lines file
instead of sending to the LLMOps API.
"""

import json
import logging
import os
from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from uipath.tracing._otel_exporters import LlmOpsHttpExporter
from uipath.tracing._utils import _SpanUtils

logger = logging.getLogger(__name__)


class LlmOpsFileExporter(SpanExporter):
    """Exports spans to a JSON Lines file in UiPath internal format.

    Uses the same format as LlmOpsHttpExporter but writes to a local file
    instead of sending to the API. Each span is written as a JSON object
    on a separate line (JSON Lines format).
    """

    def __init__(self, file_path: str, trace_id: str | None = None) -> None:
        """Initialize the file exporter.

        Args:
            file_path: Path to the output file (will be created if it doesn't exist)
            trace_id: Optional trace ID override (same as LlmOpsHttpExporter)
        """
        super().__init__()
        self.file_path = file_path
        self.trace_id = trace_id

        # Cache LlmOpsHttpExporter instance for attribute processing
        # This avoids creating a new httpx.Client for every span
        self._http_exporter = LlmOpsHttpExporter(trace_id=self.trace_id)

        # Ensure the directory exists
        dir_path = os.path.dirname(self.file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Clear the file at initialization to start fresh
        with open(self.file_path, "w") as f:
            f.write("")

        logger.info(f"LlmOps file exporter initialized: {self.file_path}")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to the JSON Lines file in UiPath internal format.

        Args:
            spans: Sequence of spans to export

        Returns:
            SpanExportResult indicating success or failure
        """
        if len(spans) == 0:
            logger.debug("No spans to export")
            return SpanExportResult.SUCCESS

        # Filter out spans marked for dropping
        filtered_spans = [s for s in spans if not self._should_drop_span(s)]

        if len(filtered_spans) == 0:
            logger.debug("No spans to export after filtering dropped spans")
            return SpanExportResult.SUCCESS

        logger.debug(f"Exporting {len(filtered_spans)} spans to {self.file_path}")

        try:
            # Convert spans using the same logic as LlmOpsHttpExporter
            span_list = [
                _SpanUtils.otel_span_to_uipath_span(
                    span, custom_trace_id=self.trace_id, serialize_attributes=False
                ).to_dict(serialize_attributes=False)
                for span in filtered_spans
            ]

            # Process span attributes (same as LlmOpsHttpExporter)
            for span_data in span_list:
                self._process_span_attributes(span_data)

            # Serialize attributes
            for span_data in span_list:
                if isinstance(span_data.get("Attributes"), dict):
                    span_data["Attributes"] = json.dumps(span_data["Attributes"])

            # Write each span as a JSON line
            with open(self.file_path, "a") as f:
                for span_data in span_list:
                    f.write(json.dumps(span_data) + "\n")

            logger.debug(f"Successfully exported {len(span_list)} spans")
            return SpanExportResult.SUCCESS

        except Exception as e:
            logger.error(f"Failed to export spans to {self.file_path}: {e}")
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the exporter.

        Args:
            timeout_millis: Timeout in milliseconds (unused for file export)

        Returns:
            Always True for file export
        """
        return True

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        logger.info(f"LlmOps file exporter shutdown: {self.file_path}")

    def _should_drop_span(self, span: ReadableSpan) -> bool:
        """Check if span is marked for dropping.

        Spans with telemetry.filter="drop" are skipped.

        Args:
            span: The span to check

        Returns:
            True if span should be dropped, False otherwise
        """
        attrs = span.attributes or {}
        return attrs.get("telemetry.filter") == "drop"

    def _process_span_attributes(self, span_data: dict[str, Any]) -> None:
        """Process span attributes using LlmOpsHttpExporter logic.

        This delegates to the LlmOpsHttpExporter's processing logic to ensure
        identical output format.

        Args:
            span_data: Span dict with Attributes as dict or JSON string
        """
        # Use cached exporter instance to avoid creating httpx.Client per span
        self._http_exporter._process_span_attributes(span_data)
