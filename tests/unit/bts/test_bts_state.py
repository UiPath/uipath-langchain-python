from uuid import uuid4

from uipath.platform.automation_tracker import OperationStatus, TransactionStatus

from uipath_agents._bts.bts_state import BtsState, ToolOperationState


def test_bts_state_defaults() -> None:
    state = BtsState()
    assert state.transaction_id is None
    assert state.transaction_created is False
    assert state.transaction_status == TransactionStatus.UNKNOWN
    assert state.agent_operation_id is None
    assert state.agent_operation_status == OperationStatus.UNKNOWN
    assert state.tool_operations == {}
    assert state.pending_tasks == set()


def test_tool_operation_state_defaults() -> None:
    op = ToolOperationState(
        operation_id="txn123-elem456",
        transaction_id="txn123",
        parent_operation_id="txn123-parent789",
        name="my_tool",
        tool_type="process",
        fingerprint="fp1",
    )
    assert op.status == OperationStatus.UNKNOWN
    assert op.result is None
    assert op.attributes == {}


def test_bts_state_track_tool_operation() -> None:
    state = BtsState()
    run_id = uuid4()
    op = ToolOperationState(
        operation_id="txn-elem",
        transaction_id="txn",
        parent_operation_id="txn-parent",
        name="tool",
        tool_type="process",
        fingerprint="fp",
    )
    state.tool_operations[run_id] = op
    assert run_id in state.tool_operations
    assert state.tool_operations[run_id].name == "tool"
