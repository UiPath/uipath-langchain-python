"""Tests for OpenTelemetry tracing configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest

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

    def test_configures_instrumentors(self):
        """Test that instrumentors are configured."""
        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "AsyncioInstrumentor") as mock_asyncio,
            patch.object(tracing, "HTTPXClientInstrumentor") as mock_httpx,
            patch.object(tracing, "AioHttpClientInstrumentor") as mock_aiohttp,
            patch.object(tracing, "SQLite3Instrumentor") as mock_sqlite,
        ):
            for mock in [mock_asyncio, mock_httpx, mock_aiohttp, mock_sqlite]:
                mock.return_value = MagicMock()

            tracing.configure_telemetry()

            for mock in [mock_asyncio, mock_httpx, mock_aiohttp, mock_sqlite]:
                mock.return_value.instrument.assert_called_once()

    def test_sets_configured_flag(self):
        """Test that configured flag is set after configuration."""
        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "AsyncioInstrumentor"),
            patch.object(tracing, "HTTPXClientInstrumentor"),
            patch.object(tracing, "AioHttpClientInstrumentor"),
            patch.object(tracing, "SQLite3Instrumentor"),
        ):
            assert not _TelemetryState.configured

            tracing.configure_telemetry()

            assert _TelemetryState.configured

    def test_idempotent_only_configures_once(self):
        """Test that configure_telemetry only runs once."""
        with (
            patch.object(tracing, "setup_otel_env") as mock_setup,
            patch.object(tracing, "AsyncioInstrumentor") as mock_asyncio,
            patch.object(tracing, "HTTPXClientInstrumentor"),
            patch.object(tracing, "AioHttpClientInstrumentor"),
            patch.object(tracing, "SQLite3Instrumentor"),
        ):
            mock_asyncio.return_value = MagicMock()

            tracing.configure_telemetry()
            tracing.configure_telemetry()  # Second call should be no-op

            mock_setup.assert_called_once()
            mock_asyncio.assert_called_once()

    def test_calls_setup_otel_env(self):
        """Test that setup_otel_env is called during configuration."""
        with (
            patch.object(tracing, "setup_otel_env") as mock_setup,
            patch.object(tracing, "AsyncioInstrumentor"),
            patch.object(tracing, "HTTPXClientInstrumentor"),
            patch.object(tracing, "AioHttpClientInstrumentor"),
            patch.object(tracing, "SQLite3Instrumentor"),
        ):
            tracing.configure_telemetry()

            mock_setup.assert_called_once()

    def test_adds_azure_exporter_when_trace_manager_provided(self):
        """Test that Azure exporter is added when trace_manager is provided."""
        mock_trace_manager = MagicMock()
        mock_exporter = MagicMock()

        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "_get_azure_exporter", return_value=mock_exporter),
            patch.object(tracing, "AsyncioInstrumentor"),
            patch.object(tracing, "HTTPXClientInstrumentor"),
            patch.object(tracing, "AioHttpClientInstrumentor"),
            patch.object(tracing, "SQLite3Instrumentor"),
        ):
            tracing.configure_telemetry(trace_manager=mock_trace_manager)

            mock_trace_manager.add_span_exporter.assert_called_once_with(mock_exporter)

    def test_skips_azure_exporter_when_no_trace_manager(self):
        """Test that Azure exporter is skipped when no trace_manager."""
        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "_get_azure_exporter") as mock_get_exporter,
            patch.object(tracing, "AsyncioInstrumentor"),
            patch.object(tracing, "HTTPXClientInstrumentor"),
            patch.object(tracing, "AioHttpClientInstrumentor"),
            patch.object(tracing, "SQLite3Instrumentor"),
        ):
            tracing.configure_telemetry(trace_manager=None)

            mock_get_exporter.assert_not_called()

    def test_skips_azure_exporter_when_not_configured(self):
        """Test that Azure exporter setup is skipped when get_azure_exporter returns None."""
        mock_trace_manager = MagicMock()

        with (
            patch.object(tracing, "setup_otel_env"),
            patch.object(tracing, "_get_azure_exporter", return_value=None),
            patch.object(tracing, "AsyncioInstrumentor"),
            patch.object(tracing, "HTTPXClientInstrumentor"),
            patch.object(tracing, "AioHttpClientInstrumentor"),
            patch.object(tracing, "SQLite3Instrumentor"),
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
