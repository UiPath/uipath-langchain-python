import json
import logging
import os
import sqlite3
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from uipath.tracing._utils import _SpanUtils

from .LangchainSpanProcessor import BaseSpanProcessor, LangchainSpanProcessor

logger = logging.getLogger(__name__)


class SqliteExporter(SpanExporter):
    """
    An exporter that writes spans to a SQLite database file.

    This exporter is useful for debugging and local development. It serializes
    the spans and inserts them into a 'spans' table in the specified database.
    """

    def __init__(self, db_path: str, processor: BaseSpanProcessor = None):
        """
        Initializes the SqliteExporter.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        self._processor = processor
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _create_table(self):
        """Creates the 'spans' table if it doesn't already exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    span_id TEXT,
                    parent_span_id TEXT,
                    name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    span_type TEXT,
                    attributes TEXT
                )
            """
            )
            conn.commit()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Exports a batch of spans to the SQLite database.

        Args:
            spans: A sequence of ReadableSpan objects.

        Returns:
            The result of the export operation.
        """
        try:
            uipath_spans = [
                _SpanUtils.otel_span_to_uipath_span(span).to_dict() for span in spans
            ]

            if self._processor:
                processed_spans = [
                    self._processor.process_span(span) for span in uipath_spans
                ]
            else:
                processed_spans = uipath_spans

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for span in processed_spans:
                    # The 'attributes' field is a JSON string, so we store it as TEXT.
                    attributes_json = span.get("attributes", "{}")
                    if not isinstance(attributes_json, str):
                        attributes_json = json.dumps(attributes_json)

                    cursor.execute(
                        """
                        INSERT INTO spans (
                            trace_id, span_id, parent_span_id, name,
                            start_time, end_time, span_type, attributes
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            span.get("TraceId"),
                            span.get("SpanId"),
                            span.get("ParentSpanId"),
                            span.get("Name"),
                            span.get("StartTime"),
                            span.get("EndTime"),
                            span.get("SpanType"),
                            attributes_json,
                        ),
                    )
                conn.commit()

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to export spans to {self.db_path}: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shuts down the exporter."""
        pass
