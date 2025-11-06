"""Tests for OpenTelemetry tracing configuration."""

import os
from unittest.mock import MagicMock, patch

from uipath_lowcode._observability import tracing


class TestGetAzureExporter:
    """Test get_azure_exporter function."""

    def test_no_connection_string_returns_none(self, caplog):
        """Test that missing connection string returns None."""
        with patch.dict(os.environ, {}, clear=True):
            result = tracing.get_azure_exporter()

            assert result is None
            assert "Azure Monitor exporter will not be configured" in caplog.text

    def test_with_connection_string_returns_exporter(self):
        """Test that valid connection string returns AzureMonitorTraceExporter."""
        conn_str = "InstrumentationKey=test-key;IngestionEndpoint=https://test.com"

        with (
            patch.dict(
                os.environ,
                {"APPLICATIONINSIGHTS_CONNECTION_STRING": conn_str},
                clear=True,
            ),
            patch(
                "uipath_lowcode._observability.tracing.AzureMonitorTraceExporter"
            ) as mock_exporter,
        ):
            mock_instance = MagicMock()
            mock_exporter.return_value = mock_instance

            result = tracing.get_azure_exporter()

            mock_exporter.assert_called_once_with(connection_string=conn_str)
            assert result == mock_instance

    def test_logs_info_when_configured(self, caplog):
        """Test that info message is logged when exporter is created."""
        conn_str = "InstrumentationKey=test-key;IngestionEndpoint=https://test.com"

        with (
            patch.dict(
                os.environ,
                {"APPLICATIONINSIGHTS_CONNECTION_STRING": conn_str},
                clear=True,
            ),
            patch("uipath_lowcode._observability.tracing.AzureMonitorTraceExporter"),
        ):
            tracing.get_azure_exporter()

            assert "Configuring Azure Monitor trace exporter" in caplog.text


class TestShutdownTelemetry:
    """Test shutdown_telemetry function."""

    def test_shutdown_when_not_initialized(self):
        """Test that shutdown is safe when OTEL was never initialized."""
        # Should not raise
        tracing.shutdown_telemetry()

    def test_shutdown_flushes_provider(self):
        """Test that shutdown flushes the tracer provider."""
        mock_provider = MagicMock()

        with patch.object(tracing.trace, "get_tracer_provider") as mock_get_provider:
            mock_get_provider.return_value = mock_provider

            tracing.shutdown_telemetry()

            mock_provider.force_flush.assert_called_once_with(timeout_millis=10000)

    def test_shutdown_handles_no_force_flush_method(self):
        """Test shutdown when provider doesn't have force_flush."""
        mock_provider = MagicMock(spec=[])  # No force_flush method

        with patch.object(tracing.trace, "get_tracer_provider") as mock_get_provider:
            mock_get_provider.return_value = mock_provider

            # Should not raise
            tracing.shutdown_telemetry()

    def test_shutdown_handles_errors_gracefully(self, caplog):
        """Test that shutdown handles errors gracefully."""
        mock_provider = MagicMock()
        mock_provider.force_flush.side_effect = Exception("Flush error")

        with patch.object(tracing.trace, "get_tracer_provider") as mock_get_provider:
            mock_get_provider.return_value = mock_provider

            # Should not raise
            tracing.shutdown_telemetry()

            assert "Error flushing telemetry" in caplog.text


class TestGetTracer:
    """Test get_tracer function."""

    def test_get_tracer_returns_tracer(self):
        """Test that get_tracer returns a tracer instance."""
        with patch.object(tracing.trace, "get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            result = tracing.get_tracer("test_module")

            mock_get_tracer.assert_called_once_with("test_module")
            assert result == mock_tracer
