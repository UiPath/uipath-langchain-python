"""Tests for LlmOpsFileExporter whitelist filtering."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from uipath_agents._observability.llmops_file_exporter import LlmOpsFileExporter


class TestLlmOpsFileExporterWhitelist:
    """Tests for whitelist-based span filtering."""

    @pytest.fixture
    def exporter(self, tmp_path: Path) -> LlmOpsFileExporter:
        return LlmOpsFileExporter(file_path=str(tmp_path / "traces.jsonl"))

    def test_keeps_span_with_custom_instrumentation_marker(
        self, exporter: LlmOpsFileExporter
    ) -> None:
        """Spans with uipath.custom_instrumentation=True should be kept."""
        span = MagicMock()
        span.attributes = {
            "uipath.custom_instrumentation": True,
            "type": "agentRun",
        }

        assert exporter._should_drop_span(span) is False

    @pytest.mark.parametrize(
        "span_type",
        [
            "agentRun",
            "llmCall",
            "completion",
            "toolCall",
            "processTool",
            "escalationTool",
            "agentOutput",
            "preGuardrails",
            "postGuardrails",
        ],
    )
    def test_keeps_all_custom_span_types(
        self, exporter: LlmOpsFileExporter, span_type: str
    ) -> None:
        """All our custom span types with marker should be kept."""
        span = MagicMock()
        span.attributes = {
            "uipath.custom_instrumentation": True,
            "type": span_type,
        }

        assert exporter._should_drop_span(span) is False

    def test_drops_span_without_marker(self, exporter: LlmOpsFileExporter) -> None:
        """Spans without uipath.custom_instrumentation marker should be dropped."""
        span = MagicMock()
        span.attributes = {"span_type": "function_call_sync"}

        assert exporter._should_drop_span(span) is True

    def test_drops_span_with_empty_attributes(
        self, exporter: LlmOpsFileExporter
    ) -> None:
        """Spans with empty attributes should be dropped."""
        span = MagicMock()
        span.attributes = {}

        assert exporter._should_drop_span(span) is True

    def test_drops_span_with_none_attributes(
        self, exporter: LlmOpsFileExporter
    ) -> None:
        """Spans with None attributes should be dropped."""
        span = MagicMock()
        span.attributes = None

        assert exporter._should_drop_span(span) is True

    def test_drops_http_span(self, exporter: LlmOpsFileExporter) -> None:
        """HTTP auto-instrumentation spans should be dropped."""
        span = MagicMock()
        span.name = "POST"
        span.attributes = {"http.method": "POST", "http.url": "https://api.example.com"}

        assert exporter._should_drop_span(span) is True

    def test_drops_openinference_span(self, exporter: LlmOpsFileExporter) -> None:
        """OpenInference spans should be dropped."""
        span = MagicMock()
        span.attributes = {"openinference.span.kind": "LLM"}

        assert exporter._should_drop_span(span) is True

    def test_drops_sdk_traced_span(self, exporter: LlmOpsFileExporter) -> None:
        """SDK @traced spans (with span_type but no marker) should be dropped."""
        span = MagicMock()
        span.attributes = {"span_type": "function_call_sync", "run_type": "uipath"}

        assert exporter._should_drop_span(span) is True

    def test_drops_span_with_false_marker(self, exporter: LlmOpsFileExporter) -> None:
        """Spans with uipath.custom_instrumentation=False should be dropped."""
        span = MagicMock()
        span.attributes = {
            "uipath.custom_instrumentation": False,
            "type": "agentRun",
        }

        assert exporter._should_drop_span(span) is True
