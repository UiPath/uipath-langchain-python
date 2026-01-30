"""Tests for PII filtering span exporter."""

import json
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from uipath_agents._observability.exporters.pii_filtering_exporter import (
    PIIFilteringExporter,
    _redact_attributes,
    _redact_value,
    _RedactedSpan,
)


@pytest.fixture
def _disable_selective_redaction():
    """Disable selective redaction for tests expecting full redaction."""
    from uipath_agents._observability.exporters import pii_filtering_exporter

    original = pii_filtering_exporter._PRESERVE_FIELDS
    pii_filtering_exporter._PRESERVE_FIELDS = frozenset()
    yield
    pii_filtering_exporter._PRESERVE_FIELDS = original


class TestRedactValue:
    """Tests for _redact_value function with full redaction (no preserve fields)."""

    @pytest.fixture(autouse=True)
    def _setup(self, _disable_selective_redaction):
        """Use full redaction for all tests in this class."""
        pass

    def test_redact_plain_string(self) -> None:
        """Test redacting a plain string."""
        result = _redact_value("sensitive data here")
        assert "REDACTED" in result
        assert "length=19" in result

    def test_redact_json_string(self) -> None:
        """Test redacting a JSON string."""
        json_str = json.dumps({"messages": [{"content": "secret"}]})
        result = _redact_value(json_str)
        assert "REDACTED" in result
        assert "dict" in result

    def test_redact_dict(self) -> None:
        """Test redacting a dictionary."""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        result = _redact_value(data)
        assert "REDACTED" in result
        assert "keys=3" in result
        assert "key1" in result or "key2" in result or "key3" in result

    def test_redact_list(self) -> None:
        """Test redacting a list."""
        result = _redact_value([1, 2, 3, 4, 5])
        assert "REDACTED" in result
        assert "length=5" in result

    def test_redact_other_types(self) -> None:
        """Test redacting other types."""
        result = _redact_value(42)
        assert "REDACTED" in result
        assert "int" in result


class TestRedactAttributes:
    """Tests for _redact_attributes function."""

    def test_redact_pii_attributes(self) -> None:
        """Test that PII attributes are redacted."""
        attrs = {
            "input.value": '{"messages": [{"content": "sensitive"}]}',
            "output.value": "response text",
            "llm.model_name": "gpt-4",  # Should be kept
        }
        redacted = _redact_attributes(attrs)

        assert "REDACTED" in str(redacted["input.value"])
        assert "REDACTED" in str(redacted["output.value"])
        assert redacted["llm.model_name"] == "gpt-4"

    def test_preserve_metadata_attributes(self) -> None:
        """Test that metadata attributes are preserved."""
        attrs: dict[str, str | int] = {
            "input.value": "sensitive",
            "llm.model_name": "gpt-4",
            "llm.token_count.total": 100,
            "openinference.span.kind": "CHAIN",
        }
        redacted = _redact_attributes(attrs)

        assert "REDACTED" in str(redacted["input.value"])
        assert redacted["llm.model_name"] == "gpt-4"
        assert redacted["llm.token_count.total"] == 100
        assert redacted["openinference.span.kind"] == "CHAIN"

    def test_empty_attributes(self) -> None:
        """Test handling of empty attributes."""
        result = _redact_attributes(None)
        assert result == {}

        result = _redact_attributes({})
        assert result == {}


class TestRedactedSpan:
    """Tests for _RedactedSpan wrapper."""

    def test_wrapper_preserves_span_properties(self) -> None:
        """Test that wrapper preserves all span properties except attributes."""
        # Create a mock span
        original_span = Mock(spec=ReadableSpan)
        original_span.name = "test_span"
        original_span.attributes = {
            "input.value": "sensitive",
            "llm.model_name": "gpt-4",
        }

        # Create redacted version
        redacted_attrs = {"input.value": "[REDACTED]", "llm.model_name": "gpt-4"}
        wrapped = _RedactedSpan(original_span, redacted_attrs)

        # Verify properties are preserved
        assert wrapped.name == "test_span"
        assert wrapped.attributes == redacted_attrs

        # Verify other properties are delegated
        assert wrapped.context == original_span.context
        assert wrapped.parent == original_span.parent


class TestPIIFilteringExporter:
    """Tests for PIIFilteringExporter."""

    def test_exporter_redacts_attributes(self) -> None:
        """Test that exporter redacts PII attributes before delegating."""
        # Create mock delegate exporter
        delegate = Mock()
        delegate.export.return_value = SpanExportResult.SUCCESS

        # Create filtering exporter
        exporter = PIIFilteringExporter(delegate)

        # Create mock spans with PII
        span1 = Mock(spec=ReadableSpan)
        span1.attributes = {
            "input.value": "sensitive user data",
            "output.value": "sensitive response",
            "llm.model_name": "gpt-4",
        }

        span2 = Mock(spec=ReadableSpan)
        span2.attributes = {
            "input.value": "more sensitive data",
        }

        # Export spans
        result = exporter.export([span1, span2])

        # Verify delegate was called
        assert result == SpanExportResult.SUCCESS
        delegate.export.assert_called_once()

        # Verify spans passed to delegate have redacted attributes
        exported_spans = delegate.export.call_args[0][0]
        assert len(exported_spans) == 2

        # Check first span
        assert "REDACTED" in str(exported_spans[0].attributes["input.value"])
        assert "REDACTED" in str(exported_spans[0].attributes["output.value"])
        assert exported_spans[0].attributes["llm.model_name"] == "gpt-4"

        # Check second span
        assert "REDACTED" in str(exported_spans[1].attributes["input.value"])

    def test_exporter_handles_no_attributes(self) -> None:
        """Test that exporter handles spans with no attributes."""
        delegate = Mock()
        delegate.export.return_value = SpanExportResult.SUCCESS

        exporter = PIIFilteringExporter(delegate)

        span = Mock(spec=ReadableSpan)
        span.attributes = None

        result = exporter.export([span])
        assert result == SpanExportResult.SUCCESS

    def test_exporter_lifecycle(self) -> None:
        """Test exporter lifecycle methods."""
        delegate = Mock()
        delegate.force_flush.return_value = True

        exporter = PIIFilteringExporter(delegate)

        # Test shutdown
        exporter.shutdown()
        delegate.shutdown.assert_called_once()

        # Test force_flush
        result = exporter.force_flush(timeout_millis=5000)
        assert result is True
        delegate.force_flush.assert_called_once_with(5000)

    def test_exporter_with_empty_span_list(self) -> None:
        """Test exporter with empty span list."""
        delegate = Mock()
        delegate.export.return_value = SpanExportResult.SUCCESS

        exporter = PIIFilteringExporter(delegate)

        result = exporter.export([])
        assert result == SpanExportResult.SUCCESS
        delegate.export.assert_called_once_with([])


class TestSelectiveRedactionModes:
    """Tests for selective vs full redaction modes."""

    def test_full_redaction_mode(self) -> None:
        """Test full redaction when preserve fields is empty."""
        from uipath_agents._observability.exporters import pii_filtering_exporter

        original = pii_filtering_exporter._PRESERVE_FIELDS
        try:
            # Set empty preserve fields = full redaction mode
            pii_filtering_exporter._PRESERVE_FIELDS = frozenset()

            attrs = {
                "input.value": json.dumps(
                    {
                        "messages": ["sensitive"],
                        "inner_state": {"data": "should also be redacted"},
                    }
                )
            }

            redacted = _redact_attributes(attrs)

            # With full redaction, everything should be redacted
            result_str = str(redacted["input.value"])
            assert "REDACTED" in result_str
            assert "dict" in result_str
            # Should not preserve inner_state in full redaction mode
            assert "inner_state" not in result_str or "REDACTED" in result_str
        finally:
            pii_filtering_exporter._PRESERVE_FIELDS = original

    def test_selective_redaction_mode(self) -> None:
        """Test selective redaction with preserve fields configured."""
        attrs = {
            "input.value": json.dumps(
                {
                    "messages": ["sensitive"],
                    "inner_state": {"data": "preserved"},
                    "user_data": {"name": "John"},
                }
            )
        }

        redacted = _redact_attributes(attrs)

        # Parse the result
        result = json.loads(redacted["input.value"])

        # messages should be redacted
        assert "REDACTED" in str(result["messages"])

        # inner_state should be preserved (in default preserve fields)
        assert result["inner_state"] == {"data": "preserved"}

        # user_data should be redacted (not in preserve fields)
        assert "REDACTED" in str(result["user_data"])

    def test_custom_preserve_fields(self) -> None:
        """Test with custom preserve fields configuration."""
        from uipath_agents._observability.exporters import pii_filtering_exporter

        original = pii_filtering_exporter._PRESERVE_FIELDS
        try:
            # Set custom preserve fields
            pii_filtering_exporter._PRESERVE_FIELDS = frozenset({"metadata", "config"})

            data = {
                "messages": ["sensitive"],
                "metadata": {"id": "123"},
                "config": {"setting": "value"},
                "user_info": {"name": "Jane"},
            }

            # Simulate this being in an attribute value
            attrs = {"input.value": json.dumps(data)}
            redacted = _redact_attributes(attrs)
            result = json.loads(redacted["input.value"])

            # messages and user_info should be redacted
            assert "REDACTED" in str(result["messages"])
            assert "REDACTED" in str(result["user_info"])

            # metadata and config should be preserved
            assert result["metadata"] == {"id": "123"}
            assert result["config"] == {"setting": "value"}
        finally:
            pii_filtering_exporter._PRESERVE_FIELDS = original


class TestIntegrationScenarios:
    """End-to-end integration tests for common scenarios."""

    def test_langgraph_agent_span(self) -> None:
        """Test typical LangGraph agent span with selective redaction."""
        attrs: dict[str, str | int] = {
            "input.value": json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello, my SSN is 123-45-6789"}
                    ],
                    "inner_state": {
                        "job_attachments": {"file1": "report.pdf"},
                        "step_count": 5,
                    },
                }
            ),
            "output.value": json.dumps(
                {
                    "messages": [
                        {"role": "assistant", "content": "I can help with that"}
                    ],
                    "inner_state": {
                        "job_attachments": {"file1": "report.pdf"},
                        "step_count": 6,
                    },
                }
            ),
            "llm.model_name": "gpt-4o",
            "llm.token_count.total": 150,
        }

        redacted = _redact_attributes(attrs)

        # Verify PII attributes are redacted
        input_result = json.loads(redacted["input.value"])
        output_result = json.loads(redacted["output.value"])

        # Messages should be redacted (contain PII)
        assert "REDACTED" in str(input_result["messages"])
        assert "REDACTED" in str(output_result["messages"])

        # inner_state should be preserved
        assert input_result["inner_state"]["job_attachments"] == {"file1": "report.pdf"}
        assert input_result["inner_state"]["step_count"] == 5
        assert output_result["inner_state"]["job_attachments"] == {
            "file1": "report.pdf"
        }
        assert output_result["inner_state"]["step_count"] == 6

        # Metadata attributes should not be touched (don't match patterns)
        assert redacted["llm.model_name"] == "gpt-4o"
        assert redacted["llm.token_count.total"] == 150

    def test_no_redaction_for_non_pii_attributes(self) -> None:
        """Test that non-PII attributes are not touched."""
        attrs: dict[str, str | int] = {
            "span.name": "agent_execution",
            "custom.metric": 42,
            "app.version": "1.0.0",
        }

        redacted = _redact_attributes(attrs)

        # All should be unchanged (no matches)
        assert redacted == attrs
