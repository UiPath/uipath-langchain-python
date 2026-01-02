"""Trace context storage for interruptible process preservation.

Provides persistence of trace context across process boundaries,
enabling resumed executions to continue the same trace.
"""

from typing import Any, Optional, Protocol, TypedDict


class TraceContextData(TypedDict):
    """Data structure for persisted trace context."""

    trace_id: str  # Hex-encoded trace ID
    span_id: str  # Hex-encoded span ID
    parent_span_id: Optional[str]  # Hex-encoded parent span ID (None for root)
    name: str  # Span name
    start_time: str  # ISO format timestamp
    attributes: dict[str, Any]  # Span attributes to restore
    # Pending interruptible tool spans (for resume without duplication)
    pending_tool_span_id: Optional[str]  # Tool call span awaiting completion
    pending_process_span_id: Optional[str]  # Process tool span awaiting completion
    pending_tool_name: Optional[str]  # Tool name for matching on resume


class TraceContextStorage(Protocol):
    """Protocol for trace context persistence across process boundaries.

    Implementations store trace context when an agent suspends (HITL/interrupt)
    and restore it when the agent resumes, allowing LLMOps to show both
    executions as part of the same trace.
    """

    async def save_trace_context(
        self, runtime_id: str, context: TraceContextData
    ) -> None:
        """Save trace context for later resume.

        Args:
            runtime_id: Unique identifier for the runtime instance
            context: Trace context data to persist
        """
        ...

    async def load_trace_context(self, runtime_id: str) -> Optional[TraceContextData]:
        """Load previously saved trace context.

        Args:
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Saved trace context if exists, None otherwise
        """
        ...

    async def clear_trace_context(self, runtime_id: str) -> None:
        """Remove trace context after successful completion.

        Args:
            runtime_id: Unique identifier for the runtime instance
        """
        ...
