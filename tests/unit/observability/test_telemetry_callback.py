"""Tests for event_emitter track_event and flush_events."""

from unittest.mock import patch

from uipath_agents._observability.event_emitter import (
    AgentRunEvent,
    flush_events,
    track_event,
)


class TestTrackEvent:
    """Test track_event functionality."""

    @patch("uipath_agents._observability.event_emitter._track_event")
    def test_track_event_calls_uipath_track_event(self, mock_track_event):
        """Test that track_event calls the uipath.telemetry track_event."""
        properties = {"key1": "value1", "key2": "value2"}

        track_event("test_event", properties)

        mock_track_event.assert_called_once_with("test_event", properties)

    @patch("uipath_agents._observability.event_emitter._track_event")
    def test_track_event_with_none_properties(self, mock_track_event):
        """Test track_event with None properties."""
        track_event("test_event", None)

        mock_track_event.assert_called_once_with("test_event", None)


class TestFlushEvents:
    """Test flush_events functionality."""

    @patch("uipath_agents._observability.event_emitter._flush_events")
    def test_flush_events_delegates(self, mock_flush):
        """Test that flush_events delegates to uipath.telemetry flush_events."""
        flush_events()

        mock_flush.assert_called_once()


class TestTelemetryEventNames:
    """Test telemetry event name constants."""

    def test_event_name_constants(self):
        """Test that event name constants are properly defined."""
        assert AgentRunEvent.STARTED == "AgentRun.Start"
        assert AgentRunEvent.COMPLETED == "AgentRun.End"
        assert AgentRunEvent.FAILED == "AgentRun.Failed"


class TestTelemetryIntegration:
    """Integration tests for telemetry functionality."""

    @patch("uipath_agents._observability.event_emitter._track_event")
    def test_end_to_end_telemetry_flow(self, mock_track_event):
        """Test complete telemetry flow through track_event to uipath.telemetry."""
        properties = {
            "AgentName": "test-agent",
            "AgentId": "test-id",
            "Model": "gpt-4",
            "Temperature": "0.7",
        }

        track_event(AgentRunEvent.STARTED, properties)

        mock_track_event.assert_called_once_with(AgentRunEvent.STARTED, properties)
