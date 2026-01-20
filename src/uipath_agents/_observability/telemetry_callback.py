"""LangChain callback handler for Application Insights telemetry.

This callback focuses solely on sending custom telemetry events to Application Insights
for agent lifecycle monitoring, using the shared telemetry client from uipath-python.
"""

import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from uipath.telemetry import flush_events as _flush_events
from uipath.telemetry import track_event as _track_event

# Suppress Application Insights HTTP request/response logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)

logger = logging.getLogger(__name__)

# Telemetry event names
AGENTRUN_STARTED = "AgentRun.Start"
AGENTRUN_COMPLETED = "AgentRun.End"
AGENTRUN_FAILED = "AgentRun.Failed"


def track_event(
    name: str,
    properties: Optional[Dict[str, Any]] = None,
    measurements: Optional[Dict[str, float]] = None,
) -> None:
    """Track a custom event to Application Insights.

    Args:
        name: Name of the event
        properties: Properties for the event (converted to strings)
        measurements: Numeric measurements for the event (currently ignored,
                     kept for backward compatibility)
    """
    logger.info(f"track_event called: {name}, properties: {properties}")
    _track_event(name, properties)
    logger.info(f"_track_event completed for: {name}")


class AppInsightsTelemetryCallback(BaseCallbackHandler):
    """LangChain callback that sends custom telemetry events to Application Insights.

    This callback is dedicated to tracking agent lifecycle events for monitoring
    and analytics purposes. It sends events directly to AppInsights without
    creating OpenTelemetry spans.

    Usage:
        callback = AppInsightsTelemetryCallback()
        callback.set_agent_info("MyAgent", "agent-123")
        runtime = SomeRuntime(callbacks=[callback])
        await runtime.execute(input)
    """

    def __init__(self) -> None:
        """Initialize the telemetry callback."""
        super().__init__()
        self._agent_name: Optional[str] = None
        self._agent_id: Optional[str] = None

    def set_agent_info(self, agent_name: str, agent_id: Optional[str] = None) -> None:
        """Set agent information for telemetry events.

        Args:
            agent_name: Name of the agent
            agent_id: Unique identifier for the agent instance
        """
        self._agent_name = agent_name
        self._agent_id = agent_id

    def track_event(
        self, name: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a custom event to Application Insights.

        Args:
            name: Name of the event
            properties: Properties for the event
        """
        track_event(name, properties)

    def cleanup(self) -> None:
        """Clean up and flush any pending telemetry events."""
        _flush_events()
