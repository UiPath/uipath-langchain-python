"""LangChain callback handler for BTS tool operation tracking."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from uipath.platform.automation_tracker import OperationStatus
from uipath.platform.common import UiPathConfig

from .bts_attributes import (
    build_agent_tool_attributes,
    build_common_operation_attributes,
    build_context_grounding_tool_attributes,
    build_escalation_tool_attributes,
    build_integration_tool_attributes,
    build_process_tool_attributes,
)
from .bts_helpers import extract_transaction_id, generate_guid, generate_operation_id
from .bts_state import BtsState, ToolOperationState

logger = logging.getLogger(__name__)


class BtsCallback(AsyncCallbackHandler):
    """Tracks BTS tool operations via LangChain callback events."""

    def __init__(self, state: BtsState) -> None:
        self._state = state

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._state.agent_operation_id or not self._state.tracker_service:
            return

        metadata = metadata or {}
        tool_type = metadata.get("tool_type", "")
        display_name = metadata.get("display_name", serialized.get("name", "unknown"))

        # Reuse operation from pre-suspend run to avoid duplicate startOperation
        resume_key = (tool_type, display_name)
        resumed_op = self._state.resumed_tool_operations.pop(resume_key, None)
        if resumed_op:
            self._state.tool_operations[run_id] = resumed_op
            return

        transaction_id = extract_transaction_id(self._state.agent_operation_id)
        operation_id = generate_operation_id(transaction_id)
        fingerprint = generate_guid()

        attributes = self._build_tool_attributes(tool_type, metadata)

        bts_context = metadata.get("_bts_context")

        op = ToolOperationState(
            operation_id=operation_id,
            transaction_id=transaction_id,
            parent_operation_id=self._state.agent_operation_id,
            name=display_name,
            tool_type=tool_type,
            fingerprint=fingerprint,
            attributes=attributes,
            bts_context_ref=bts_context,
        )
        self._state.tool_operations[run_id] = op

        if bts_context is not None:
            bts_context["parent_operation_id"] = op.operation_id

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        op = self._state.tool_operations.pop(run_id, None)
        if not op or not self._state.tracker_service:
            return

        self._apply_dynamic_attributes(op)

        op.status = OperationStatus.SUCCESSFUL
        self._state.ended_tool_operations.append(op)

        self._fire_start_operation(op)
        self._fire_end_operation(op)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        op = self._state.tool_operations.pop(run_id, None)
        if not op or not self._state.tracker_service:
            return

        self._apply_dynamic_attributes(op)

        # GraphInterrupt = suspend signal — fire startOperation, skip endOperation
        if self._is_graph_interrupt(error):
            self._fire_start_operation(op)
            self._state.ended_tool_operations.append(op)
            return

        op.status = OperationStatus.FAILED
        op.result = str(error)
        self._state.ended_tool_operations.append(op)

        self._fire_start_operation(op)
        self._fire_end_operation(op)

    def _build_tool_attributes(
        self,
        tool_type: str,
        metadata: Dict[str, Any],
    ) -> dict[str, str]:
        """Build type-specific BTS attributes from tool metadata."""
        job_key = UiPathConfig.job_key
        if tool_type == "process":
            return build_process_tool_attributes(
                job_key=job_key,
                tool_type=tool_type,
                process_name=metadata.get("display_name"),
            )
        elif tool_type == "agent":
            return build_agent_tool_attributes(
                job_key=job_key,
                tool_type=tool_type,
            )
        elif tool_type == "context":
            return build_context_grounding_tool_attributes(
                job_key=job_key,
                tool_type=tool_type,
                index_name=metadata.get("index_name"),
                context_retrieval_mode=metadata.get("context_retrieval_mode"),
            )
        elif tool_type == "escalation":
            return build_escalation_tool_attributes(
                job_key=job_key,
                tool_type=tool_type,
            )
        elif tool_type == "integration":
            return build_integration_tool_attributes(
                job_key=job_key,
                tool_type=tool_type,
                connector_key=metadata.get("connector_key"),
                connector_name=metadata.get("connector_name"),
            )
        else:
            return build_common_operation_attributes(
                job_key=job_key,
                tool_type=tool_type,
            )

    # --- BTS context → attribute mapping ---
    _DYNAMIC_ATTR_MAP: dict[str, str] = {
        "task_key": "XUiPathTaskKey",
        "wait_for_job_key": "XUiPathWaitForJobKey",
        "wait_for_agent_job_key": "XUiPathWaitForAgentJobKey",
        "index_id": "XUiPathIndexId",
    }

    @staticmethod
    def _apply_dynamic_attributes(op: ToolOperationState) -> None:
        """Read dynamic values from the tool's _bts_context and update op attributes."""
        ctx = op.bts_context_ref
        if not ctx:
            return
        for ctx_key, attr_name in BtsCallback._DYNAMIC_ATTR_MAP.items():
            value = ctx.get(ctx_key)
            if value is not None:
                op.attributes[attr_name] = str(value)

    def _fire_start_operation(self, op: ToolOperationState) -> None:
        """Fire the deferred startOperation HTTP call (skips if already fired)."""
        if op.start_operation_fired or not self._state.tracker_service:
            return
        op.start_operation_fired = True
        self._fire_and_forget(
            self._state.tracker_service.start_operation_async(
                transaction_id=op.transaction_id,
                operation_id=op.operation_id,
                name=op.name,
                fingerprint=op.fingerprint,
                parent_operation=op.parent_operation_id,
                attributes=op.attributes,
            )
        )

    def _fire_end_operation(self, op: ToolOperationState) -> None:
        """Fire the endOperation HTTP call."""
        if not self._state.tracker_service:
            return
        self._fire_and_forget(
            self._state.tracker_service.end_operation_async(
                transaction_id=op.transaction_id,
                operation_id=op.operation_id,
                name=op.name,
                fingerprint=op.fingerprint,
                parent_operation=op.parent_operation_id,
                status=op.status,
                result=op.result,
                attributes=op.attributes,
            )
        )

    @staticmethod
    def _is_graph_interrupt(error: BaseException) -> bool:
        """Check if the error is a GraphInterrupt (suspend signal)."""
        return type(error).__name__ == "GraphInterrupt"

    def _fire_and_forget(self, coro: Any) -> None:
        """Launch a coroutine as a background task."""
        task = asyncio.get_running_loop().create_task(self._safe_execute(coro))
        self._state.pending_tasks.add(task)
        task.add_done_callback(self._state.pending_tasks.discard)

    @staticmethod
    async def _safe_execute(coro: Any) -> None:
        try:
            await coro
        except Exception:
            logger.warning("BTS tool operation call failed", exc_info=True)
