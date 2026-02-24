"""SQLite-based trace context storage implementation.

Adapts SqliteResumableStorage for trace context persistence,
using the key-value storage interface from uipath-langchain-python.
"""

from typing import TYPE_CHECKING, Any, Optional

from .trace_context_storage import TraceContextData, TraceContextStorage

if TYPE_CHECKING:
    from uipath_langchain.runtime.storage import SqliteResumableStorage


class SqliteTraceContextStorage(TraceContextStorage):
    """Adapts SqliteResumableStorage for trace context persistence.

    Uses the async key-value storage interface (set_value/get_value) from
    uipath-langchain-python PR #372 to persist trace context across
    process boundaries.

    Example:
        storage = SqliteResumableStorage(memory)
        trace_storage = SqliteTraceContextStorage(storage)

        # On suspend
        await trace_storage.save_trace_context(runtime_id, context_data)

        # On resume
        saved = await trace_storage.load_trace_context(runtime_id)

        # On completion
        await trace_storage.clear_trace_context(runtime_id)
    """

    NAMESPACE = "trace_context"
    KEY = "agent_span"

    def __init__(self, storage: "SqliteResumableStorage"):
        """Initialize with underlying SQLite storage.

        Args:
            storage: SqliteResumableStorage instance with key-value support
        """
        self._storage = storage

    async def save_trace_context(
        self, runtime_id: str, context: TraceContextData
    ) -> None:
        """Save trace context for later resume.

        Args:
            runtime_id: Unique identifier for the runtime instance
            context: Trace context data to persist
        """
        await self._storage.set_value(
            runtime_id=runtime_id,
            namespace=self.NAMESPACE,
            key=self.KEY,
            value=dict(context),  # Convert TypedDict to regular dict for JSON
        )

    async def load_trace_context(self, runtime_id: str) -> Optional[TraceContextData]:
        """Load previously saved trace context.

        Args:
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Saved trace context if exists, None otherwise
        """
        value: Optional[dict[str, Any]] = await self._storage.get_value(
            runtime_id=runtime_id,
            namespace=self.NAMESPACE,
            key=self.KEY,
        )

        if value is None:
            return None

        # Reconstruct TypedDict from stored dict
        return TraceContextData(
            trace_id=value["trace_id"],
            span_id=value["span_id"],
            parent_span_id=value.get("parent_span_id"),
            name=value["name"],
            start_time=value["start_time"],
            start_time_ns=value.get("start_time_ns", 0),
            attributes=value.get("attributes", {}),
            pending_tool_span_id=value.get("pending_tool_span_id"),
            pending_process_span_id=value.get("pending_process_span_id"),
            pending_tool_name=value.get("pending_tool_name"),
            pending_tool_span=value.get("pending_tool_span"),
            pending_process_span=value.get("pending_process_span"),
            pending_escalation_span=value.get("pending_escalation_span"),
            pending_guardrail_hitl_evaluation_span=value.get(
                "pending_guardrail_hitl_evaluation_span"
            ),
            pending_guardrail_hitl_container_span=value.get(
                "pending_guardrail_hitl_container_span"
            ),
            pending_llm_span=value.get("pending_llm_span"),
        )

    async def clear_trace_context(self, runtime_id: str) -> None:
        """Remove trace context after successful completion.

        Args:
            runtime_id: Unique identifier for the runtime instance
        """
        await self._storage.set_value(
            runtime_id=runtime_id,
            namespace=self.NAMESPACE,
            key=self.KEY,
            value=None,
        )
