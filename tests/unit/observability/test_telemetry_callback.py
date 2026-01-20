"""Tests for AppInsightsTelemetryCallback and telemetry functionality."""

from unittest.mock import patch

from uipath_agents._observability.telemetry_callback import (
    AGENTRUN_COMPLETED,
    AGENTRUN_FAILED,
    AGENTRUN_STARTED,
    AppInsightsTelemetryCallback,
    track_event,
)


class TestTrackEvent:
    """Test track_event functionality."""

    @patch("uipath_agents._observability.telemetry_callback._track_event")
    def test_track_event_calls_uipath_track_event(self, mock_track_event):
        """Test that track_event calls the uipath.telemetry track_event."""
        properties = {"key1": "value1", "key2": "value2"}

        track_event("test_event", properties)

        mock_track_event.assert_called_once_with("test_event", properties)

    @patch("uipath_agents._observability.telemetry_callback._track_event")
    def test_track_event_with_measurements_ignores_measurements(self, mock_track_event):
        """Test that measurements parameter is ignored (backward compatibility)."""
        properties = {"key": "value"}
        measurements = {"metric1": 1.5}

        track_event("test_event", properties, measurements)

        # measurements should be ignored, only name and properties passed
        mock_track_event.assert_called_once_with("test_event", properties)

    @patch("uipath_agents._observability.telemetry_callback._track_event")
    def test_track_event_with_none_properties(self, mock_track_event):
        """Test track_event with None properties."""
        track_event("test_event", None)

        mock_track_event.assert_called_once_with("test_event", None)


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

    def test_set_agent_info_without_id(self):
        """Test setting agent information without agent ID."""
        callback = AppInsightsTelemetryCallback()

        callback.set_agent_info("test-agent")

        assert callback._agent_name == "test-agent"
        assert callback._agent_id is None

    @patch("uipath_agents._observability.telemetry_callback.track_event")
    def test_track_event_calls_global_function(self, mock_track_event):
        """Test that callback track_event calls the global track_event function."""
        callback = AppInsightsTelemetryCallback()
        properties = {"key": "value"}

        callback.track_event("test_event", properties)

        mock_track_event.assert_called_once_with("test_event", properties)

    @patch("uipath_agents._observability.telemetry_callback._flush_events")
    def test_cleanup_flushes_events(self, mock_flush_events):
        """Test that cleanup method flushes pending telemetry events."""
        callback = AppInsightsTelemetryCallback()

        callback.cleanup()

        mock_flush_events.assert_called_once()


class TestTelemetryEventNames:
    """Test telemetry event name constants."""

    def test_event_name_constants(self):
        """Test that event name constants are properly defined."""
        assert AGENTRUN_STARTED == "AgentRun.Start"
        assert AGENTRUN_COMPLETED == "AgentRun.End"
        assert AGENTRUN_FAILED == "AgentRun.Failed"


class TestTelemetryIntegration:
    """Integration tests for telemetry functionality."""

    @patch("uipath_agents._observability.telemetry_callback._track_event")
    def test_end_to_end_telemetry_flow(self, mock_track_event):
        """Test complete telemetry flow from callback to uipath.telemetry."""
        callback = AppInsightsTelemetryCallback()
        callback.set_agent_info("test-agent", "test-id")

        properties = {
            "AgentName": "test-agent",
            "AgentId": "test-id",
            "Model": "gpt-4",
            "Temperature": "0.7",
        }

        callback.track_event(AGENTRUN_STARTED, properties)

        # Verify track_event was called with correct arguments
        mock_track_event.assert_called_once_with(AGENTRUN_STARTED, properties)
