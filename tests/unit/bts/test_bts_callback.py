import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from uipath.platform.automation_tracker import (
    AutomationTrackerService,
    OperationStatus,
)

from uipath_agents._bts.bts_callback import BtsCallback
from uipath_agents._bts.bts_state import BtsState, ToolOperationState


@pytest.fixture
def mock_tracker() -> AsyncMock:
    tracker = AsyncMock(spec=AutomationTrackerService)
    tracker._organization_id = "org-1"
    tracker._tenant_id = "tenant-1"
    return tracker


@pytest.fixture
def active_state(mock_tracker: AsyncMock) -> BtsState:
    state = BtsState()
    state.transaction_id = "txn123"
    state.agent_operation_id = "txn123-agentop456"
    state.tracker_service = mock_tracker
    return state


@pytest.mark.asyncio
async def test_on_tool_start_skips_when_bts_inactive() -> None:
    state = BtsState()
    callback = BtsCallback(state)
    await callback.on_tool_start(
        serialized={"name": "my_tool"},
        input_str="{}",
        run_id=uuid4(),
        metadata={"tool_type": "process", "display_name": "MyProcess"},
    )
    assert len(state.tool_operations) == 0


@pytest.mark.asyncio
async def test_on_tool_start_creates_tool_operation(active_state: BtsState) -> None:
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "my_tool"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "process", "display_name": "MyProcess"},
    )
    assert run_id in active_state.tool_operations
    op = active_state.tool_operations[run_id]
    assert op.name == "MyProcess"
    assert op.tool_type == "process"
    assert op.parent_operation_id == "txn123-agentop456"
    assert op.transaction_id == "txn123"
    assert op.status == OperationStatus.UNKNOWN


@pytest.mark.asyncio
async def test_on_tool_end_fires_start_and_end_operation(
    active_state: BtsState,
) -> None:
    """startOperation is deferred until on_tool_end, then both start and end fire."""
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "process", "display_name": "P"},
    )

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    tracker.start_operation_async.assert_not_called()

    await callback.on_tool_end(output="result", run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker.start_operation_async.assert_called_once()
    tracker.end_operation_async.assert_called_once()
    call_kwargs = tracker.end_operation_async.call_args.kwargs
    assert call_kwargs["status"] == OperationStatus.SUCCESSFUL
    assert run_id not in active_state.tool_operations
    assert len(active_state.ended_tool_operations) == 1


@pytest.mark.asyncio
async def test_on_tool_error_fires_start_and_end_operation(
    active_state: BtsState,
) -> None:
    """Real errors fire both deferred startOperation and endOperation as FAILED."""
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "process", "display_name": "P"},
    )
    await callback.on_tool_error(error=ValueError("boom"), run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    tracker.start_operation_async.assert_called_once()
    tracker.end_operation_async.assert_called_once()
    call_kwargs = tracker.end_operation_async.call_args.kwargs
    assert call_kwargs["status"] == OperationStatus.FAILED
    assert run_id not in active_state.tool_operations
    assert len(active_state.ended_tool_operations) == 1
    assert active_state.ended_tool_operations[0].result == "boom"


@pytest.mark.asyncio
async def test_on_tool_end_ignores_unknown_run_id(active_state: BtsState) -> None:
    callback = BtsCallback(active_state)
    await callback.on_tool_end(output="result", run_id=uuid4())
    assert len(active_state.pending_tasks) == 0
    assert len(active_state.ended_tool_operations) == 0


@pytest.mark.asyncio
async def test_on_tool_error_fires_start_only_on_graph_interrupt(
    active_state: BtsState,
) -> None:
    """GraphInterrupt fires deferred startOperation but skips endOperation."""
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "escalation", "display_name": "CreateTask"},
    )

    class GraphInterrupt(Exception):
        pass

    await callback.on_tool_error(error=GraphInterrupt("suspend"), run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    tracker.start_operation_async.assert_called_once()
    tracker.end_operation_async.assert_not_called()
    assert run_id not in active_state.tool_operations
    assert len(active_state.ended_tool_operations) == 1
    op = active_state.ended_tool_operations[0]
    assert op.status == OperationStatus.UNKNOWN
    assert op.result is None


@pytest.mark.asyncio
async def test_on_tool_error_fires_start_and_end_for_real_errors(
    active_state: BtsState,
) -> None:
    """Non-GraphInterrupt errors fire both startOperation and endOperation as FAILED."""
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "process", "display_name": "P"},
    )
    await callback.on_tool_error(error=RuntimeError("connection lost"), run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    tracker.start_operation_async.assert_called_once()
    tracker.end_operation_async.assert_called_once()
    call_kwargs = tracker.end_operation_async.call_args.kwargs
    assert call_kwargs["status"] == OperationStatus.FAILED
    assert active_state.ended_tool_operations[0].result == "connection lost"


@pytest.mark.asyncio
async def test_on_tool_start_includes_job_key_and_tool_type(
    active_state: BtsState,
) -> None:
    """Tool attributes should contain XUiPathSourceKey and XUiPathToolType."""
    callback = BtsCallback(active_state)
    run_id = uuid4()
    with patch("uipath_agents._bts.bts_callback.UiPathConfig") as mock_config:
        mock_config.job_key = "job-abc"
        await callback.on_tool_start(
            serialized={"name": "t"},
            input_str="{}",
            run_id=run_id,
            metadata={"tool_type": "process", "display_name": "MyProc"},
        )
    op = active_state.tool_operations[run_id]
    assert op.attributes["XUiPathSourceKey"] == "job-abc"
    assert op.attributes["XUiPathToolType"] == "process"


@pytest.mark.asyncio
async def test_on_tool_start_reuses_resumed_operation(
    active_state: BtsState,
) -> None:
    """When a resumed operation exists for (tool_type, name), reuse it."""
    resumed_op = ToolOperationState(
        operation_id="op-prev",
        transaction_id="txn123",
        parent_operation_id="txn123-agentop456",
        name="MyProc",
        tool_type="process",
        fingerprint="fp-prev",
        attributes={"XUiPathToolType": "process"},
    )
    active_state.resumed_tool_operations[("process", "MyProc")] = resumed_op

    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "process", "display_name": "MyProc"},
    )

    assert active_state.tool_operations[run_id] is resumed_op
    assert ("process", "MyProc") not in active_state.resumed_tool_operations
    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    tracker.start_operation_async.assert_not_called()


@pytest.mark.asyncio
async def test_on_tool_start_no_resume_for_different_tool(
    active_state: BtsState,
) -> None:
    """Resumed operations only match on (tool_type, name)."""
    resumed_op = ToolOperationState(
        operation_id="op-prev",
        transaction_id="txn123",
        parent_operation_id="txn123-agentop456",
        name="OtherProc",
        tool_type="process",
        fingerprint="fp-prev",
    )
    active_state.resumed_tool_operations[("process", "OtherProc")] = resumed_op

    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "process", "display_name": "MyProc"},
    )

    op = active_state.tool_operations[run_id]
    assert op is not resumed_op
    assert op.name == "MyProc"
    # startOperation is deferred — not called during on_tool_start
    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    tracker.start_operation_async.assert_not_called()


@pytest.mark.asyncio
async def test_resumed_op_end_fires_end_operation(
    active_state: BtsState,
) -> None:
    """After resume dedup, on_tool_end should fire endOperation normally."""
    resumed_op = ToolOperationState(
        operation_id="op-prev",
        transaction_id="txn123",
        parent_operation_id="txn123-agentop456",
        name="CreateTask",
        tool_type="escalation",
        fingerprint="fp-prev",
        attributes={"XUiPathToolType": "escalation"},
        start_operation_fired=True,
    )
    active_state.resumed_tool_operations[("escalation", "CreateTask")] = resumed_op

    callback = BtsCallback(active_state)
    run_id = uuid4()

    # on_tool_start reuses the resumed op
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={"tool_type": "escalation", "display_name": "CreateTask"},
    )

    # on_tool_end fires endOperation with the original operation IDs
    await callback.on_tool_end(output="task completed", run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    # startOperation was already fired before suspend — should not fire again
    tracker.start_operation_async.assert_not_called()
    tracker.end_operation_async.assert_called_once()
    call_kwargs = tracker.end_operation_async.call_args.kwargs
    assert call_kwargs["operation_id"] == "op-prev"
    assert call_kwargs["fingerprint"] == "fp-prev"
    assert call_kwargs["status"] == OperationStatus.SUCCESSFUL


@pytest.mark.asyncio
async def test_on_tool_start_writes_parent_operation_id_to_bts_context(
    active_state: BtsState,
) -> None:
    """on_tool_start writes the tool's operation_id as parent_operation_id into _bts_context."""
    bts_context: dict[str, Any] = {}
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={
            "tool_type": "process",
            "display_name": "P",
            "_bts_context": bts_context,
        },
    )
    op = active_state.tool_operations[run_id]
    assert bts_context["parent_operation_id"] == op.operation_id


@pytest.mark.asyncio
async def test_dynamic_attributes_applied_on_tool_end(
    active_state: BtsState,
) -> None:
    """Dynamic values written to _bts_context appear in attributes on both start and end."""
    bts_context: dict[str, Any] = {}
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={
            "tool_type": "process",
            "display_name": "P",
            "_bts_context": bts_context,
        },
    )

    # Simulate tool writing dynamic value (e.g., job key after invoke_async)
    bts_context["wait_for_job_key"] = "job-key-123"

    await callback.on_tool_end(output="done", run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    start_attrs = tracker.start_operation_async.call_args.kwargs["attributes"]
    end_attrs = tracker.end_operation_async.call_args.kwargs["attributes"]
    assert start_attrs["XUiPathWaitForJobKey"] == "job-key-123"
    assert end_attrs["XUiPathWaitForJobKey"] == "job-key-123"


@pytest.mark.asyncio
async def test_dynamic_attributes_applied_on_graph_interrupt(
    active_state: BtsState,
) -> None:
    """Dynamic values written before GraphInterrupt appear in startOperation attributes."""
    bts_context: dict[str, Any] = {}
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={
            "tool_type": "escalation",
            "display_name": "CreateTask",
            "_bts_context": bts_context,
        },
    )

    # Simulate tool writing task_key before GraphInterrupt
    bts_context["task_key"] = "10544662"

    class GraphInterrupt(Exception):
        pass

    await callback.on_tool_error(error=GraphInterrupt("suspend"), run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    start_attrs = tracker.start_operation_async.call_args.kwargs["attributes"]
    assert start_attrs["XUiPathTaskKey"] == "10544662"
    tracker.end_operation_async.assert_not_called()


@pytest.mark.asyncio
async def test_dynamic_attributes_multiple_keys(
    active_state: BtsState,
) -> None:
    """All mapped _bts_context keys are applied as attributes."""
    bts_context: dict[str, Any] = {}
    callback = BtsCallback(active_state)
    run_id = uuid4()
    await callback.on_tool_start(
        serialized={"name": "t"},
        input_str="{}",
        run_id=run_id,
        metadata={
            "tool_type": "agent",
            "display_name": "SubAgent",
            "_bts_context": bts_context,
        },
    )

    bts_context["wait_for_agent_job_key"] = "agent-job-456"

    await callback.on_tool_end(output="done", run_id=run_id)
    await asyncio.gather(*active_state.pending_tasks, return_exceptions=True)

    tracker = active_state.tracker_service
    assert isinstance(tracker, AsyncMock)
    end_attrs = tracker.end_operation_async.call_args.kwargs["attributes"]
    assert end_attrs["XUiPathWaitForAgentJobKey"] == "agent-job-456"
