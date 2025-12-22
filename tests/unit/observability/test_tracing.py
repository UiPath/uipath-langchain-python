"""Tests for OpenTelemetry tracing configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from uipath_agents._observability import tracing
from uipath_agents._observability.tracing import _TelemetryState


@pytest.fixture(autouse=True)
def reset_telemetry_state():
    """Reset telemetry state before and after each test."""
    _TelemetryState.configured = False
    _TelemetryState.instrumentors = []
    yield
    _TelemetryState.configured = False
    _TelemetryState.instrumentors = []


class TestConfigureTelemetry:
    """Test configure_telemetry function."""

    def test_sets_configured_flag(self):
        """Test that configured flag is set after configuration."""
        with patch.object(tracing, "setup_otel_env"):
            assert not _TelemetryState.configured

            tracing.configure_telemetry()

            assert _TelemetryState.configured

    def test_idempotent_only_configures_once(self):
        """Test that configure_telemetry only runs once."""
        with patch.object(tracing, "setup_otel_env") as mock_setup:
            tracing.configure_telemetry()
            tracing.configure_telemetry()  # Second call should be no-op

            mock_setup.assert_called_once()

    def test_calls_setup_otel_env(self):
        """Test that setup_otel_env is called during configuration."""
        with patch.object(tracing, "setup_otel_env") as mock_setup:
            tracing.configure_telemetry()

            mock_setup.assert_called_once()

    def test_adds_filtered_azure_exporter_when_trace_manager_provided(self):
        """Test that filtered Azure exporter is added when trace_manager is provided."""
        mock_trace_manager = MagicMock()
        mock_exporter = MagicMock()

        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "_get_azure_exporter", return_value=mock_exporter),
        ):
            tracing.configure_telemetry(trace_manager=mock_trace_manager)

            # Should wrap Azure exporter with FilteringSpanExporter
            mock_trace_manager.add_span_exporter.assert_called_once()
            added_exporter = mock_trace_manager.add_span_exporter.call_args[0][0]
            assert isinstance(added_exporter, tracing.FilteringSpanExporter)
            assert added_exporter._delegate is mock_exporter
            assert added_exporter._filter_fn is tracing.is_openinference_span

    def test_skips_azure_exporter_when_no_trace_manager(self):
        """Test that Azure exporter is skipped when no trace_manager."""
        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "_get_azure_exporter") as mock_get_exporter,
        ):
            tracing.configure_telemetry(trace_manager=None)

            mock_get_exporter.assert_not_called()

    def test_skips_azure_exporter_when_not_configured(self):
        """Test that Azure exporter setup is skipped when get_azure_exporter returns None."""
        mock_trace_manager = MagicMock()

        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "_get_azure_exporter", return_value=None),
        ):
            tracing.configure_telemetry(trace_manager=mock_trace_manager)

            mock_trace_manager.add_span_exporter.assert_not_called()


class TestGetAzureExporter:
    """Test _get_azure_exporter function."""

    def test_no_connection_string_returns_none(self, caplog):
        """Test that missing connection string returns None."""
        with patch.dict(os.environ, {}, clear=True):
            result = tracing._get_azure_exporter()

            assert result is None
            assert "Azure Monitor exporter not configured" in caplog.text

    def test_with_connection_string_returns_exporter(self):
        """Test that valid connection string returns AzureMonitorTraceExporter."""
        conn_str = "InstrumentationKey=test-key;IngestionEndpoint=https://test.com"

        with (
            patch.dict(
                os.environ,
                {"APPLICATIONINSIGHTS_CONNECTION_STRING": conn_str},
                clear=True,
            ),
            patch.object(tracing, "AzureMonitorTraceExporter") as mock_exporter,
        ):
            mock_instance = MagicMock()
            mock_exporter.return_value = mock_instance

            result = tracing._get_azure_exporter()

            mock_exporter.assert_called_once_with(connection_string=conn_str)
            assert result == mock_instance


class TestFilteringSpanExporter:
    """Test FilteringSpanExporter."""

    def test_export_filters_spans(self):
        """Test that only matching spans are exported."""
        mock_delegate = MagicMock()
        mock_delegate.export.return_value = SpanExportResult.SUCCESS

        def filter_fn(s):
            return s.name == "keep"

        exporter = tracing.FilteringSpanExporter(mock_delegate, filter_fn)

        span_keep = MagicMock()
        span_keep.name = "keep"
        span_drop = MagicMock()
        span_drop.name = "drop"

        result = exporter.export([span_keep, span_drop])

        assert result == SpanExportResult.SUCCESS
        mock_delegate.export.assert_called_once_with([span_keep])

    def test_export_returns_success_when_all_filtered(self):
        """Test SUCCESS is returned when all spans are filtered out."""
        mock_delegate = MagicMock()
        exporter = tracing.FilteringSpanExporter(mock_delegate, lambda s: False)

        span = MagicMock()
        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        mock_delegate.export.assert_not_called()

    def test_shutdown_delegates(self):
        """Test shutdown is delegated."""
        mock_delegate = MagicMock()
        exporter = tracing.FilteringSpanExporter(mock_delegate, lambda s: True)

        exporter.shutdown()

        mock_delegate.shutdown.assert_called_once()

    def test_force_flush_delegates(self):
        """Test force_flush is delegated."""
        mock_delegate = MagicMock()
        mock_delegate.force_flush.return_value = True
        exporter = tracing.FilteringSpanExporter(mock_delegate, lambda s: True)

        result = exporter.force_flush(5000)

        assert result is True
        mock_delegate.force_flush.assert_called_once_with(5000)


class TestIsOpeninferenceSpan:
    """Test is_openinference_span function."""

    def test_returns_true_for_openinference_scope(self):
        """Test returns True for openinference.* scope names."""
        span = MagicMock()
        span.instrumentation_scope.name = "openinference.instrumentation.langchain"

        assert tracing.is_openinference_span(span) is True

    def test_returns_false_for_non_openinference_scope(self):
        """Test returns False for non-openinference scope names."""
        span = MagicMock()
        span.instrumentation_scope.name = "uipath.agents.tracing"

        assert tracing.is_openinference_span(span) is False

    def test_returns_false_when_no_scope(self):
        """Test returns False when instrumentation_scope is None."""
        span = MagicMock()
        span.instrumentation_scope = None

        assert tracing.is_openinference_span(span) is False


class TestShutdownTelemetry:
    """Test shutdown_telemetry function.

    Note: shutdown_telemetry only uninstruments libraries. It does NOT flush
    or shutdown the TracerProvider - that's the trace_manager's responsibility.
    """

    def test_noop_when_not_configured(self):
        """Test that shutdown is a no-op when not configured."""
        _TelemetryState.configured = False
        mock_instrumentor = MagicMock()
        _TelemetryState.instrumentors = [mock_instrumentor]

        tracing.shutdown_telemetry()

        # Should not uninstrument if not configured
        mock_instrumentor.uninstrument.assert_not_called()

    def test_uninstruments_all_instrumentors(self):
        """Test that shutdown uninstruments all stored instrumentors."""
        mock_instrumentor1 = MagicMock()
        mock_instrumentor2 = MagicMock()
        _TelemetryState.configured = True
        _TelemetryState.instrumentors = [mock_instrumentor1, mock_instrumentor2]

        tracing.shutdown_telemetry()

        mock_instrumentor1.uninstrument.assert_called_once()
        mock_instrumentor2.uninstrument.assert_called_once()

    def test_resets_state_after_shutdown(self):
        """Test that state is reset after shutdown."""
        _TelemetryState.configured = True
        _TelemetryState.instrumentors = [MagicMock()]

        tracing.shutdown_telemetry()

        assert not _TelemetryState.configured
        assert _TelemetryState.instrumentors == []

    def test_handles_uninstrument_errors_gracefully(self, caplog):
        """Test that uninstrument errors are logged but don't stop shutdown."""
        mock_instrumentor = MagicMock()
        mock_instrumentor.uninstrument.side_effect = Exception("Uninstrument error")
        _TelemetryState.configured = True
        _TelemetryState.instrumentors = [mock_instrumentor]

        tracing.shutdown_telemetry()

        assert "Failed to uninstrument" in caplog.text
        assert not _TelemetryState.configured
