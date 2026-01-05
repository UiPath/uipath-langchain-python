"""LangChain callback handler for Application Insights telemetry.

This callback focuses solely on sending custom telemetry events to Application Insights
for agent lifecycle monitoring, using the official Microsoft Application Insights SDK
to ensure events appear in the customEvents table.
"""

import logging
import os
from typing import Any, Dict, Optional

# Note: TelemetryClient is imported-untyped because it is not typed, as it was
# deprecated in favor of the OpenTelemetry SDK. We are still using it because it
# is the only way to send telemetry to Application Insights.
from applicationinsights import TelemetryClient  # type: ignore[import-untyped]
from langchain_core.callbacks import BaseCallbackHandler

try:
    _HAS_APPINSIGHTS = True
except ImportError:
    _HAS_APPINSIGHTS = False
    TelemetryClient = None

# Suppress Application Insights HTTP request/response logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)

logger = logging.getLogger(__name__)

# Telemetry event names
AGENTRUN_STARTED = "AgentRun.Start.URT"
AGENTRUN_COMPLETED = "AgentRun.End.URT"
AGENTRUN_FAILED = "AgentRun.Failed.URT"


def _parse_connection_string(connection_string: str) -> Optional[str]:
    """Parse Azure Application Insights connection string to get instrumentation key."""
    try:
        parts = {}
        for part in connection_string.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                parts[key] = value
        return parts.get("InstrumentationKey")
    except Exception:
        return None


class _TelemetryClient:
    """Microsoft Application Insights SDK-based custom events client."""

    _initialized = False
    _enabled = os.getenv("UIPATH_TELEMETRY_ENABLED", "true").lower() == "true"
    _client = None

    @staticmethod
    def _initialize():
        """Initialize Application Insights telemetry client."""
        if _TelemetryClient._initialized:
            return

        _TelemetryClient._initialized = True

        if not _TelemetryClient._enabled or not _HAS_APPINSIGHTS:
            return

        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if not connection_string:
            return

        try:
            instrumentation_key = _parse_connection_string(connection_string)

            if not instrumentation_key:
                return

            _TelemetryClient._client = TelemetryClient(instrumentation_key)

        except Exception:
            # Silently fail - telemetry should never break the main application
            pass

    @staticmethod
    def track_event(
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        measurements: Optional[Dict[str, float]] = None,
    ):
        """Track a custom event using Microsoft Application Insights SDK."""
        if not _TelemetryClient._enabled:
            return

        _TelemetryClient._initialize()

        if not _TelemetryClient._client:
            return

        try:
            safe_properties = {}
            if properties:
                for key, value in properties.items():
                    if value is not None:
                        safe_properties[key] = str(value)

            _TelemetryClient._client.track_event(
                name=name, properties=safe_properties, measurements=measurements or {}
            )

            _TelemetryClient._client.flush()

        except Exception:
            pass


def track_event(
    name: str,
    properties: Optional[Dict[str, Any]] = None,
    measurements: Optional[Dict[str, float]] = None,
) -> None:
    """Track a custom event to Application Insights.

    Args:
        name: Name of the event
        properties: Properties for the event (converted to strings)
        measurements: Numeric measurements for the event
    """
    try:
        _TelemetryClient.track_event(
            name=name, properties=properties, measurements=measurements
        )
    except Exception:
        pass


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
        """Clean up any remaining state."""
        pass
