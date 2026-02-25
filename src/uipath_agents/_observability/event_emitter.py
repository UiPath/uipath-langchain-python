"""Telemetry event emitter for Application Insights.

Sends custom telemetry events (not spans) to Application Insights
for agent lifecycle monitoring, using the shared telemetry client from uipath-python.
"""

import logging
from enum import StrEnum
from typing import Any, Dict, Optional

from opentelemetry import trace
from uipath.telemetry import flush_events as _flush_events
from uipath.telemetry import track_event as _track_event

# Suppress Application Insights HTTP request/response logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)

logger = logging.getLogger(__name__)


class AgentRunEvent(StrEnum):
    """Telemetry event names for agent run lifecycle."""

    STARTED = "AgentRun.Start"
    COMPLETED = "AgentRun.End"
    FAILED = "AgentRun.Failed"


class GuardrailEvent(StrEnum):
    """Telemetry event names for guardrail actions."""

    SKIPPED = "Guardrail.Skipped"
    BLOCKED = "Guardrail.Blocked"
    LOGGED = "Guardrail.Logged"
    FILTERED = "Guardrail.Filtered"
    ESCALATED = "Guardrail.Escalated"
    ESCALATION_APPROVED = "Guardrail.EscalationApproved"
    ESCALATION_REJECTED = "Guardrail.EscalationRejected"


def track_event(
    name: str,
    properties: Optional[Dict[str, Any]] = None,
    measurements: Optional[Dict[str, float]] = None,
) -> None:
    """Track a custom event to Application Insights with trace context.

    Links the custom event to the current OpenTelemetry trace by setting
    operation_Id (trace ID) and operation_ParentId (span ID) on the event.

    Args:
        name: Name of the event
        properties: Properties for the event (converted to strings)
        measurements: Numeric measurements for the event (currently ignored,
                     kept for backward compatibility)
    """
    logger.info("track_event called: %s, properties: %s", name, properties)

    # Set operation context from current span before tracking event
    _set_operation_context_from_current_span()

    _track_event(name, properties)
    logger.info("_track_event completed for: %s", name)


def _set_operation_context_from_current_span() -> None:
    """Set Application Insights operation context from current OpenTelemetry span.

    This links custom events to the current trace by setting:
    - operation.id = trace_id (32-char hex string)
    - operation.parent_id = span_id in W3C format (|trace_id.span_id.)
    """
    try:
        # Get current span from OpenTelemetry context
        current_span = trace.get_current_span()
        if not current_span:
            return

        span_context = current_span.get_span_context()
        if not span_context or not span_context.is_valid:
            return

        # Format IDs for Application Insights
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")

        # W3C trace-context format for parent_id: |trace_id.span_id.
        operation_parent_id = f"|{trace_id}.{span_id}."

        # Access the Application Insights client and set operation context
        from uipath.telemetry._track import _AppInsightsEventClient

        client = _AppInsightsEventClient._client
        if client:
            client.context.operation.id = trace_id
            client.context.operation.parent_id = operation_parent_id

    except Exception as e:
        # Silently fail - telemetry should never break the application
        logger.warning(
            f"Failed to set operation context from current span: {type(e).__name__}: {e}"
        )


class TelemetryEventEmitter:
    """Emits custom telemetry events to Application Insights.

    Dedicated to tracking agent lifecycle events for monitoring and analytics.
    Sends events directly to AppInsights without creating OpenTelemetry spans.

    Usage:
        emitter = TelemetryEventEmitter()
        emitter.set_agent_info("MyAgent", "agent-123")
        emitter.track_event("AgentRun.Start", {"AgentName": "MyAgent"})
    """

    def __init__(self) -> None:
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
