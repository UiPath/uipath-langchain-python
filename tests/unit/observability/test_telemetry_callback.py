"""Tests for AppInsightsTelemetryCallback and telemetry functionality."""

import os
from unittest.mock import MagicMock, patch

from uipath_agents._observability.telemetry_callback import (
    AGENTRUN_COMPLETED,
    AGENTRUN_FAILED,
    AGENTRUN_STARTED,
    AppInsightsTelemetryCallback,
    _parse_connection_string,
    _TelemetryClient,
    track_event,
)


class TestConnectionStringParsing:
    """Test connection string parsing functionality."""

    def test_parse_valid_connection_string(self):
        """Test parsing a valid Application Insights connection string."""
        connection_string = "InstrumentationKey=test-key;IngestionEndpoint=https://example.com/;LiveEndpoint=https://live.example.com/"

        result = _parse_connection_string(connection_string)

        assert result == "test-key"

    def test_parse_connection_string_missing_instrumentation_key(self):
        """Test parsing connection string without InstrumentationKey."""
        connection_string = "IngestionEndpoint=https://example.com/;LiveEndpoint=https://live.example.com/"

        result = _parse_connection_string(connection_string)

        assert result is None

    def test_parse_malformed_connection_string(self):
        """Test parsing malformed connection string."""
        connection_string = "not-a-valid-connection-string"

        result = _parse_connection_string(connection_string)

        assert result is None

    def test_parse_empty_connection_string(self):
        """Test parsing empty connection string."""
        result = _parse_connection_string("")

        assert result is None


class TestTelemetryClient:
    """Test _TelemetryClient functionality."""

    def setup_method(self):
        """Reset TelemetryClient state before each test."""
        _TelemetryClient._initialized = False
        _TelemetryClient._enabled = True
        _TelemetryClient._client = None

    @patch.dict(os.environ, {"UIPATH_TELEMETRY_ENABLED": "true"})
    @patch("uipath_agents._observability.telemetry_callback._HAS_APPINSIGHTS", True)
    @patch("uipath_agents._observability.telemetry_callback.TelemetryClient")
    def test_initialize_creates_client(self, mock_telemetry_client_class):
        """Test that _initialize creates Application Insights client."""
        mock_client = MagicMock()
        mock_telemetry_client_class.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test-key;IngestionEndpoint=https://example.com/"
            },
        ):
            _TelemetryClient._initialize()

        assert _TelemetryClient._initialized is True
        assert _TelemetryClient._client is mock_client
        mock_telemetry_client_class.assert_called_once_with("test-key")

    @patch.dict(os.environ, {"UIPATH_TELEMETRY_ENABLED": "false"})
    def test_initialize_disabled_telemetry(self):
        """Test that initialization is skipped when telemetry is disabled."""
        _TelemetryClient._enabled = False

        _TelemetryClient._initialize()

        assert _TelemetryClient._initialized is True
        assert _TelemetryClient._client is None

    @patch.dict(os.environ, {}, clear=True)
    @patch("uipath_agents._observability.telemetry_callback._HAS_APPINSIGHTS", True)
    def test_initialize_no_connection_string(self):
        """Test initialization when no connection string is provided."""
        _TelemetryClient._initialize()

        assert _TelemetryClient._initialized is True
        assert _TelemetryClient._client is None

    @patch("uipath_agents._observability.telemetry_callback._HAS_APPINSIGHTS", False)
    def test_initialize_no_applicationinsights_package(self):
        """Test initialization when applicationinsights package is not available."""
        _TelemetryClient._initialize()

        assert _TelemetryClient._initialized is True
        assert _TelemetryClient._client is None

    @patch.dict(os.environ, {"UIPATH_TELEMETRY_ENABLED": "true"})
    @patch("uipath_agents._observability.telemetry_callback._HAS_APPINSIGHTS", True)
    @patch("uipath_agents._observability.telemetry_callback.TelemetryClient")
    def test_track_event_calls_client(self, mock_telemetry_client_class):
        """Test that track_event calls the Application Insights client."""
        mock_client = MagicMock()
        mock_telemetry_client_class.return_value = mock_client
        _TelemetryClient._client = mock_client
        _TelemetryClient._initialized = True

        properties = {"key1": "value1", "key2": 123, "key3": None}
        measurements = {"metric1": 1.5}

        _TelemetryClient.track_event("test_event", properties, measurements)

        mock_client.track_event.assert_called_once_with(
            name="test_event",
            properties={
                "key1": "value1",
                "key2": "123",
            },  # None values filtered, others converted to strings
            measurements={"metric1": 1.5},
        )
        mock_client.flush.assert_called_once()

    def test_track_event_disabled(self):
        """Test that track_event does nothing when telemetry is disabled."""
        _TelemetryClient._enabled = False

        # Should not raise any exception
        _TelemetryClient.track_event("test_event", {"key": "value"})

    @patch(
        "uipath_agents._observability.telemetry_callback._TelemetryClient.track_event"
    )
    def test_global_track_event_function(self, mock_track_event):
        """Test the global track_event function calls _TelemetryClient.track_event."""
        properties = {"key": "value"}
        measurements = {"metric": 1.0}

        track_event("test_event", properties, measurements)

        mock_track_event.assert_called_once_with(
            name="test_event", properties=properties, measurements=measurements
        )


class TestAppInsightsTelemetryCallback:
    """Test AppInsightsTelemetryCallback LangChain callback handler."""

    def test_init_creates_callback(self):
        """Test that callback can be initialized."""
        callback = AppInsightsTelemetryCallback()

        assert callback._agent_name is None
        assert callback._agent_id is None

    def test_set_agent_info(self):
        """Test setting agent information."""
        callback = AppInsightsTelemetryCallback()

        callback.set_agent_info("test-agent", "test-id")

        assert callback._agent_name == "test-agent"
        assert callback._agent_id == "test-id"

    @patch("uipath_agents._observability.telemetry_callback.track_event")
    def test_track_event_calls_global_function(self, mock_track_event):
        """Test that callback track_event calls the global track_event function."""
        callback = AppInsightsTelemetryCallback()
        properties = {"key": "value"}

        callback.track_event("test_event", properties)

        mock_track_event.assert_called_once_with("test_event", properties)

    def test_cleanup_does_not_raise(self):
        """Test that cleanup method does not raise exceptions."""
        callback = AppInsightsTelemetryCallback()

        # Should not raise any exception
        callback.cleanup()


class TestTelemetryEventNames:
    """Test telemetry event name constants."""

    def test_event_name_constants(self):
        """Test that event name constants are properly defined."""
        assert AGENTRUN_STARTED == "AgentRun.Start.URT"
        assert AGENTRUN_COMPLETED == "AgentRun.End.URT"
        assert AGENTRUN_FAILED == "AgentRun.Failed.URT"


class TestTelemetryIntegration:
    """Integration tests for telemetry functionality."""

    def setup_method(self):
        """Reset TelemetryClient state before each test."""
        _TelemetryClient._initialized = False
        _TelemetryClient._enabled = True
        _TelemetryClient._client = None

    @patch.dict(os.environ, {"UIPATH_TELEMETRY_ENABLED": "true"})
    @patch("uipath_agents._observability.telemetry_callback._HAS_APPINSIGHTS", True)
    @patch("uipath_agents._observability.telemetry_callback.TelemetryClient")
    def test_end_to_end_telemetry_flow(self, mock_telemetry_client_class):
        """Test complete telemetry flow from callback to Application Insights."""
        mock_client = MagicMock()
        mock_telemetry_client_class.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test-key;IngestionEndpoint=https://example.com/"
            },
        ):
            callback = AppInsightsTelemetryCallback()
            callback.set_agent_info("test-agent", "test-id")

            properties = {
                "AgentName": "test-agent",
                "AgentId": "test-id",
                "Model": "gpt-4",
                "Temperature": "0.7",
            }

            callback.track_event(AGENTRUN_STARTED, properties)

        # Verify client was initialized and called
        mock_telemetry_client_class.assert_called_once_with("test-key")
        mock_client.track_event.assert_called_once_with(
            name=AGENTRUN_STARTED, properties=properties, measurements={}
        )
        mock_client.flush.assert_called_once()

    @patch.dict(os.environ, {"UIPATH_TELEMETRY_ENABLED": "false"})
    def test_telemetry_disabled_no_tracking(self):
        """Test that no tracking occurs when telemetry is disabled."""
        callback = AppInsightsTelemetryCallback()

        # Should not raise any exception even with invalid properties
        callback.track_event("test_event", None)
