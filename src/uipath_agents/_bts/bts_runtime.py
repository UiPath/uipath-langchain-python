"""BTS runtime wrapper for transaction and agent operation lifecycle.

Manages transaction create-or-reuse, agent operation start/end,
suspend/resume state persistence, and background HTTP task draining.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from uipath.platform.automation_tracker import OperationStatus, TransactionStatus
from uipath.platform.common import UiPathConfig
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeProtocol,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.events import UiPathRuntimeEvent
from uipath.runtime.schema import UiPathRuntimeSchema

from .bts_attributes import (
    build_agent_operation_attributes,
    build_transaction_attributes,
)
from .bts_callback import BtsCallback
from .bts_helpers import (
    extract_transaction_id,
    generate_guid,
    generate_operation_id,
)
from .bts_state import BtsState, ToolOperationState
from .bts_storage import SqliteBtsStateStorage

logger = logging.getLogger(__name__)


class BtsRuntime:
    """Wrapper that adds BTS transaction/operation tracking to any runtime.

    Manages transaction create-or-reuse, agent operation start/end,
    state persistence across suspend/resume, and drains background
    HTTP tasks on dispose.
    """

    def __init__(
        self,
        delegate: UiPathRuntimeProtocol,
        state: BtsState,
        callback: BtsCallback,
        agent_name: str,
        runtime_id: str,
        bts_storage: Optional[SqliteBtsStateStorage] = None,
        parent_operation_id: Optional[str] = None,
    ) -> None:
        self._delegate = delegate
        self._state = state
        self._callback = callback
        self._agent_name = agent_name
        self._runtime_id = runtime_id
        self._bts_storage = bts_storage
        self._parent_op_id: Optional[str] = parent_operation_id

    @property
    def delegate(self) -> UiPathRuntimeProtocol:
        """The wrapped runtime delegate."""
        return self._delegate

    async def execute(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the delegate with BTS transaction/operation tracking."""
        await self._init_or_resume()
        try:
            result = await self._delegate.execute(input, options)
        except Exception as e:
            self._set_failed(e)
            await self._finalize()
            raise

        await self._handle_result(result)
        return result

    async def stream(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream the delegate with BTS transaction/operation tracking."""
        await self._init_or_resume()
        final_result: Optional[UiPathRuntimeResult] = None
        try:
            async for event in self._delegate.stream(input, options):
                if isinstance(event, UiPathRuntimeResult):
                    final_result = event
                yield event
        except Exception as e:
            self._set_failed(e)
            await self._finalize()
            raise

        if final_result:
            await self._handle_result(final_result)

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Passthrough schema for the delegate."""
        return await self._delegate.get_schema()

    async def dispose(self) -> None:
        """Dispose the delegate."""
        await self._delegate.dispose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    # --- Private lifecycle methods ---

    async def _init_or_resume(self) -> None:
        """Check for saved state (resume) or initialize fresh."""
        if self._bts_storage:
            saved = await self._bts_storage.load_bts_state(self._runtime_id)
            if saved:
                self._restore_state(saved)
                return

        self._init_transaction_and_operation()

    def _restore_state(self, saved: dict[str, Any]) -> None:
        """Restore BTS state from a previously saved dict."""
        self._state.transaction_id = saved["transaction_id"]
        self._state.transaction_created = saved["transaction_created"]
        self._state.transaction_name = saved.get("transaction_name")
        self._state.transaction_fingerprint = saved.get("transaction_fingerprint")
        self._state.agent_operation_id = saved["agent_operation_id"]
        self._state.agent_operation_fingerprint = saved.get(
            "agent_operation_fingerprint"
        )
        self._parent_op_id = saved.get("parent_operation_id")

        saved_ops = saved.get("tool_operations", [])
        for op_data in saved_ops:
            key = (op_data["tool_type"], op_data["name"])
            self._state.resumed_tool_operations[key] = ToolOperationState(
                operation_id=op_data["operation_id"],
                transaction_id=op_data["transaction_id"],
                parent_operation_id=op_data["parent_operation_id"],
                name=op_data["name"],
                tool_type=op_data["tool_type"],
                fingerprint=op_data["fingerprint"],
                attributes=op_data.get("attributes", {}),
                start_operation_fired=True,
            )

    def _init_transaction_and_operation(self) -> None:
        """Create or reuse transaction, then start agent operation."""
        tracker = self._state.tracker_service
        if not tracker:
            return

        if self._parent_op_id:
            self._state.transaction_id = extract_transaction_id(self._parent_op_id)
            self._state.transaction_created = False
        else:
            self._state.transaction_id = generate_guid()
            self._state.transaction_created = True
            self._state.transaction_fingerprint = generate_guid()
            self._state.transaction_name = self._agent_name

            self._callback._fire_and_forget(
                tracker.start_transaction_async(
                    transaction_id=self._state.transaction_id,
                    name=self._agent_name,
                    reference=self._state.transaction_id,
                    fingerprint=self._state.transaction_fingerprint,
                    attributes=build_transaction_attributes(),
                )
            )

        self._state.agent_operation_id = generate_operation_id(
            self._state.transaction_id
        )
        self._state.agent_operation_fingerprint = generate_guid()

        self._callback._fire_and_forget(
            tracker.start_operation_async(
                transaction_id=self._state.transaction_id,
                operation_id=self._state.agent_operation_id,
                name=self._agent_name,
                fingerprint=self._state.agent_operation_fingerprint,
                parent_operation=self._parent_op_id,
                attributes=build_agent_operation_attributes(
                    job_key=UiPathConfig.job_key,
                    process_name=self._agent_name,
                    process_key=UiPathConfig.process_uuid,
                    package_version="",
                    package_id="",
                ),
            )
        )

    def _set_failed(self, error: Exception) -> None:
        """Set transaction and agent operation to FAILED."""
        error_result = str(error)
        self._state.transaction_status = TransactionStatus.FAILED
        self._state.transaction_result = error_result
        self._state.agent_operation_status = OperationStatus.FAILED
        self._state.agent_operation_result = error_result

    async def _handle_result(self, result: UiPathRuntimeResult) -> None:
        """Route to suspend-save or finalize based on result status."""
        if result.status == UiPathRuntimeStatus.SUSPENDED:
            await self._save_and_suspend()
        else:
            await self._finalize()
            if self._bts_storage:
                await self._bts_storage.clear_bts_state(self._runtime_id)

    async def _save_and_suspend(self) -> None:
        """Save BTS state for later resume without ending operations."""
        if self._bts_storage:
            # Serialize ended tool ops for resume dedup
            serialized_ops: list[dict[str, Any]] = []
            for op in self._state.ended_tool_operations:
                serialized_ops.append(
                    {
                        "operation_id": op.operation_id,
                        "transaction_id": op.transaction_id,
                        "parent_operation_id": op.parent_operation_id,
                        "name": op.name,
                        "tool_type": op.tool_type,
                        "fingerprint": op.fingerprint,
                        "attributes": op.attributes,
                    }
                )

            state_dict: dict[str, Any] = {
                "transaction_id": self._state.transaction_id,
                "transaction_created": self._state.transaction_created,
                "transaction_name": self._state.transaction_name,
                "transaction_fingerprint": self._state.transaction_fingerprint,
                "agent_operation_id": self._state.agent_operation_id,
                "agent_operation_fingerprint": self._state.agent_operation_fingerprint,
                "parent_operation_id": self._parent_op_id,
                "tool_operations": serialized_ops,
            }
            await self._bts_storage.save_bts_state(self._runtime_id, state_dict)
        await self._drain_pending_tasks()

    async def _finalize(self) -> None:
        """End transaction and agent operation, then drain background tasks."""
        try:
            tracker = self._state.tracker_service
            if not tracker:
                return

            if self._state.agent_operation_status == OperationStatus.UNKNOWN:
                self._state.agent_operation_status = OperationStatus.SUCCESSFUL

            self._callback._fire_and_forget(
                tracker.end_operation_async(
                    transaction_id=self._state.transaction_id,
                    operation_id=self._state.agent_operation_id,
                    name=self._agent_name,
                    fingerprint=self._state.agent_operation_fingerprint,
                    status=self._state.agent_operation_status,
                    result=self._state.agent_operation_result,
                    attributes=build_agent_operation_attributes(
                        job_key=UiPathConfig.job_key,
                        process_name=self._agent_name,
                        process_key=UiPathConfig.process_uuid,
                        package_version="",
                        package_id="",
                    ),
                )
            )

            if self._state.transaction_created:
                if self._state.transaction_status == TransactionStatus.UNKNOWN:
                    self._state.transaction_status = TransactionStatus.SUCCESSFUL

                self._callback._fire_and_forget(
                    tracker.end_transaction_async(
                        transaction_id=self._state.transaction_id,
                        name=self._agent_name,
                        reference=self._state.transaction_id,
                        fingerprint=self._state.transaction_fingerprint,
                        status=self._state.transaction_status,
                        result=self._state.transaction_result,
                        attributes=build_transaction_attributes(),
                    )
                )

            await self._drain_pending_tasks()
        except Exception:
            logger.warning("BTS finalization failed", exc_info=True)

    async def _drain_pending_tasks(self) -> None:
        """Wait for all pending BTS background tasks to complete."""
        if self._state.pending_tasks:
            await asyncio.gather(*self._state.pending_tasks, return_exceptions=True)
