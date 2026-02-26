"""Tests for BtsRuntime transaction/operation lifecycle."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from uipath.platform.automation_tracker import (
    AutomationTrackerService,
    OperationStatus,
    TransactionStatus,
)
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus

from uipath_agents._bts.bts_callback import BtsCallback
from uipath_agents._bts.bts_runtime import BtsRuntime
from uipath_agents._bts.bts_state import BtsState
from uipath_agents._bts.bts_storage import SqliteBtsStateStorage


@pytest.fixture
def mock_delegate() -> AsyncMock:
    delegate = AsyncMock()
    delegate.execute = AsyncMock(
        return_value=UiPathRuntimeResult(
            status=UiPathRuntimeStatus.SUCCESSFUL, output={"result": "ok"}
        )
    )
    delegate.get_schema = AsyncMock(return_value=MagicMock())
    delegate.dispose = AsyncMock()
    return delegate


@pytest.fixture
def mock_tracker() -> AsyncMock:
    tracker = AsyncMock(spec=AutomationTrackerService)
    tracker._organization_id = "org-1"
    tracker._tenant_id = "tenant-1"
    return tracker


@pytest.fixture
def mock_bts_storage() -> AsyncMock:
    storage = AsyncMock(spec=SqliteBtsStateStorage)
    storage.load_bts_state = AsyncMock(return_value=None)
    storage.save_bts_state = AsyncMock()
    storage.clear_bts_state = AsyncMock()
    return storage


@pytest.fixture
def bts_state(mock_tracker: AsyncMock) -> BtsState:
    state = BtsState()
    state.tracker_service = mock_tracker
    return state


@pytest.mark.asyncio
async def test_top_level_creates_transaction(
    mock_delegate: AsyncMock,
    mock_tracker: AsyncMock,
    mock_bts_storage: AsyncMock,
    bts_state: BtsState,
) -> None:
    callback = BtsCallback(bts_state)
    runtime = BtsRuntime(
        delegate=mock_delegate,
        state=bts_state,
        callback=callback,
        agent_name="TestAgent",
        bts_storage=mock_bts_storage,
        runtime_id="rt-1",
    )
    result = await runtime.execute({"input": "test"})

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert bts_state.transaction_created is True
    mock_tracker.start_transaction_async.assert_called_once()
    mock_tracker.start_operation_async.assert_called_once()
    mock_tracker.end_operation_async.assert_called_once()
    mock_tracker.end_transaction_async.assert_called_once()
    end_op_kwargs = mock_tracker.end_operation_async.call_args.kwargs
    assert end_op_kwargs["status"] == OperationStatus.SUCCESSFUL
    end_txn_kwargs = mock_tracker.end_transaction_async.call_args.kwargs
    assert end_txn_kwargs["status"] == TransactionStatus.SUCCESSFUL
    mock_bts_storage.clear_bts_state.assert_called_once_with("rt-1")


@pytest.mark.asyncio
async def test_nested_reuses_transaction(
    mock_delegate: AsyncMock,
    mock_tracker: AsyncMock,
    mock_bts_storage: AsyncMock,
    bts_state: BtsState,
) -> None:
    parent_op = "abc123def456abc123def456abc123de-parentelem1234567890abcdef12345"
    callback = BtsCallback(bts_state)
    runtime = BtsRuntime(
        delegate=mock_delegate,
        state=bts_state,
        callback=callback,
        agent_name="ChildAgent",
        bts_storage=mock_bts_storage,
        runtime_id="rt-2",
        parent_operation_id=parent_op,
    )
    result = await runtime.execute({"input": "test"})

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert bts_state.transaction_created is False
    assert bts_state.transaction_id == "abc123def456abc123def456abc123de"
    mock_tracker.start_transaction_async.assert_not_called()
    mock_tracker.end_transaction_async.assert_not_called()
    mock_tracker.start_operation_async.assert_called_once()
    mock_tracker.end_operation_async.assert_called_once()


@pytest.mark.asyncio
async def test_execute_error_sets_failed(
    mock_delegate: AsyncMock,
    mock_tracker: AsyncMock,
    mock_bts_storage: AsyncMock,
    bts_state: BtsState,
) -> None:
    mock_delegate.execute = AsyncMock(side_effect=RuntimeError("agent failed"))
    callback = BtsCallback(bts_state)
    runtime = BtsRuntime(
        delegate=mock_delegate,
        state=bts_state,
        callback=callback,
        agent_name="FailAgent",
        bts_storage=mock_bts_storage,
        runtime_id="rt-3",
    )
    with pytest.raises(RuntimeError, match="agent failed"):
        await runtime.execute({"input": "test"})

    end_op_kwargs = mock_tracker.end_operation_async.call_args.kwargs
    assert end_op_kwargs["status"] == OperationStatus.FAILED
    end_txn_kwargs = mock_tracker.end_transaction_async.call_args.kwargs
    assert end_txn_kwargs["status"] == TransactionStatus.FAILED


@pytest.mark.asyncio
async def test_suspended_saves_state_and_skips_finalize(
    mock_delegate: AsyncMock,
    mock_tracker: AsyncMock,
    mock_bts_storage: AsyncMock,
    bts_state: BtsState,
) -> None:
    mock_delegate.execute = AsyncMock(
        return_value=UiPathRuntimeResult(
            status=UiPathRuntimeStatus.SUSPENDED, output={"interrupt": "hitl"}
        )
    )
    callback = BtsCallback(bts_state)
    runtime = BtsRuntime(
        delegate=mock_delegate,
        state=bts_state,
        callback=callback,
        agent_name="SuspendAgent",
        bts_storage=mock_bts_storage,
        runtime_id="rt-suspend",
    )
    result = await runtime.execute({"input": "test"})

    assert result.status == UiPathRuntimeStatus.SUSPENDED
    mock_tracker.start_transaction_async.assert_called_once()
    mock_tracker.start_operation_async.assert_called_once()
    mock_tracker.end_operation_async.assert_not_called()
    mock_tracker.end_transaction_async.assert_not_called()
    mock_bts_storage.save_bts_state.assert_called_once()
    saved_call = mock_bts_storage.save_bts_state.call_args
    assert saved_call.args[0] == "rt-suspend"
    saved_dict = saved_call.args[1]
    assert saved_dict["transaction_id"] == bts_state.transaction_id
    assert saved_dict["transaction_created"] is True
    assert saved_dict["agent_operation_id"] == bts_state.agent_operation_id


@pytest.mark.asyncio
async def test_resume_restores_state_and_skips_creation(
    mock_delegate: AsyncMock,
    mock_tracker: AsyncMock,
    mock_bts_storage: AsyncMock,
    bts_state: BtsState,
) -> None:
    saved_state = {
        "transaction_id": "saved_txn_id",
        "transaction_created": True,
        "transaction_name": "ResumeAgent",
        "transaction_fingerprint": "saved_fp1",
        "agent_operation_id": "saved_txn_id-saved_op",
        "agent_operation_fingerprint": "saved_fp2",
        "parent_operation_id": None,
    }
    mock_bts_storage.load_bts_state = AsyncMock(return_value=saved_state)

    callback = BtsCallback(bts_state)
    runtime = BtsRuntime(
        delegate=mock_delegate,
        state=bts_state,
        callback=callback,
        agent_name="ResumeAgent",
        bts_storage=mock_bts_storage,
        runtime_id="rt-resume",
    )
    result = await runtime.execute({"input": "test"})

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert bts_state.transaction_id == "saved_txn_id"
    assert bts_state.agent_operation_id == "saved_txn_id-saved_op"
    assert bts_state.transaction_created is True
    mock_tracker.start_transaction_async.assert_not_called()
    mock_tracker.start_operation_async.assert_not_called()
    mock_tracker.end_operation_async.assert_called_once()
    mock_tracker.end_transaction_async.assert_called_once()
    mock_bts_storage.clear_bts_state.assert_called_once_with("rt-resume")


@pytest.mark.asyncio
async def test_no_storage_still_works(
    mock_delegate: AsyncMock,
    mock_tracker: AsyncMock,
    bts_state: BtsState,
) -> None:
    callback = BtsCallback(bts_state)
    runtime = BtsRuntime(
        delegate=mock_delegate,
        state=bts_state,
        callback=callback,
        agent_name="NoStorageAgent",
        bts_storage=None,
        runtime_id="rt-nostorage",
    )
    result = await runtime.execute({"input": "test"})

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    mock_tracker.start_transaction_async.assert_called_once()
    mock_tracker.end_transaction_async.assert_called_once()
