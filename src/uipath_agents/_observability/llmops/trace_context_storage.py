"""Trace context storage for interruptible process preservation.

Provides persistence of trace context across process boundaries,
enabling resumed executions to continue the same trace.
"""

from typing import Any, Optional, Protocol, TypedDict


class PendingSpanData(TypedDict, total=False):
    """Data for a pending span that needs completion on resume."""

    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time_ns: int  # Nanoseconds since epoch
    attributes: dict[str, Any]


class TraceContextData(TypedDict):
    """Data structure for persisted trace context."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: str
    attributes: dict[str, Any]
    # Pending interruptible tool spans
    pending_tool_span_id: Optional[str]
    pending_process_span_id: Optional[str]
    pending_tool_name: Optional[str]
    pending_tool_span: Optional[PendingSpanData]
    pending_process_span: Optional[PendingSpanData]
    # Pending guardrail escalation span (for HITL)
    pending_escalation_span: Optional[PendingSpanData]
    pending_guardrail_hitl_evaluation_span: Optional[PendingSpanData]
    pending_guardrail_hitl_container_span: Optional[PendingSpanData]
    pending_llm_span: Optional[PendingSpanData]


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
