"""Tests for span processor module."""

from unittest.mock import MagicMock, patch

import pytest

from uipath_agents._observability.span_processor import (
    SourceMarkerProcessor,
    _is_sdk_internal_span,
)


class TestIsSdkInternalSpan:
    """Tests for SDK internal span detection via run_type attribute."""

    def test_returns_true_for_uipath_run_type(self) -> None:
        span = MagicMock()
        span.attributes = {"run_type": "uipath"}
        assert _is_sdk_internal_span(span) is True

    def test_returns_false_for_other_run_type(self) -> None:
        span = MagicMock()
        span.attributes = {"run_type": "other"}
        assert _is_sdk_internal_span(span) is False

    def test_returns_false_for_missing_run_type(self) -> None:
        span = MagicMock()
        span.attributes = {}
        assert _is_sdk_internal_span(span) is False

    def test_returns_false_for_none_attributes(self) -> None:
        span = MagicMock()
        span.attributes = None
        # attributes.get() will fail, but real spans always have attributes dict
        span.attributes = {}
        assert _is_sdk_internal_span(span) is False


class TestSourceMarkerProcessor:
    """Tests for SourceMarkerProcessor span filtering."""

    @pytest.fixture
    def processor(self) -> SourceMarkerProcessor:
        return SourceMarkerProcessor()

    @pytest.fixture
    def mock_span(self) -> MagicMock:
        span = MagicMock()
        span.name = "some_span"
        span.attributes = {}
        return span

    def test_marks_openinference_span_for_drop(
        self, processor: SourceMarkerProcessor
    ) -> None:
        span = MagicMock()
        span.name = "regular_span"
        span.attributes = {}

        with patch(
            "uipath_agents._observability.span_processor.is_openinference_span",
            return_value=True,
        ):
            processor.on_start(span, None)

        span.set_attribute.assert_called_once_with("telemetry.filter", "drop")

    def test_marks_sdk_internal_span_for_drop(
        self, processor: SourceMarkerProcessor
    ) -> None:
        span = MagicMock()
        span.name = "tasks_create"
        span.attributes = {"run_type": "uipath"}

        with patch(
            "uipath_agents._observability.span_processor.is_openinference_span",
            return_value=False,
        ):
            processor.on_start(span, None)

        span.set_attribute.assert_called_once_with("telemetry.filter", "drop")

    def test_does_not_mark_regular_span(
        self, processor: SourceMarkerProcessor, mock_span: MagicMock
    ) -> None:
        with patch(
            "uipath_agents._observability.span_processor.is_openinference_span",
            return_value=False,
        ):
            processor.on_start(mock_span, None)

        mock_span.set_attribute.assert_not_called()

    @pytest.mark.parametrize(
        "span_name",
        ["tasks_create", "tasks_retrieve", "processes_invoke", "llm_chat_completions"],
    )
    def test_marks_spans_with_uipath_run_type(
        self, processor: SourceMarkerProcessor, span_name: str
    ) -> None:
        span = MagicMock()
        span.name = span_name
        span.attributes = {"run_type": "uipath"}

        with patch(
            "uipath_agents._observability.span_processor.is_openinference_span",
            return_value=False,
        ):
            processor.on_start(span, None)

        span.set_attribute.assert_called_once_with("telemetry.filter", "drop")

    def test_does_not_mark_span_without_uipath_run_type(
        self, processor: SourceMarkerProcessor
    ) -> None:
        span = MagicMock()
        span.name = "tasks_create"
        span.attributes = {"run_type": "custom"}

        with patch(
            "uipath_agents._observability.span_processor.is_openinference_span",
            return_value=False,
        ):
            processor.on_start(span, None)

        span.set_attribute.assert_not_called()

    def test_on_end_is_noop(self, processor: SourceMarkerProcessor) -> None:
        span = MagicMock()
        # Should not raise
        processor.on_end(span)

    def test_shutdown_is_noop(self, processor: SourceMarkerProcessor) -> None:
        # Should not raise
        processor.shutdown()

    def test_force_flush_returns_true(self, processor: SourceMarkerProcessor) -> None:
        assert processor.force_flush() is True
