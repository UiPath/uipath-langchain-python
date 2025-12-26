import os
import tempfile
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from uipath.core.errors import ErrorCategory, UiPathPendingTriggerError
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathResumableRuntime,
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
    UiPathRuntimeStatus,
)

from uipath_langchain.runtime import UiPathLangGraphRuntime
from uipath_langchain.runtime.storage import SqliteResumableStorage


class MockTriggerHandler:
    """Mock implementation of UiPathResumeTriggerHandler."""

    def __init__(self):
        self.call_count = 0

    async def create_trigger(self, suspend_value: Any) -> UiPathResumeTrigger:
        """Create a trigger from suspend value."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            payload=suspend_value,
        )
        return trigger

    async def read_trigger(self, trigger: UiPathResumeTrigger) -> Any:
        """Read trigger and return mock response.

        1st call: success
        2nd call: fail
        3rd call: fail
        4th call: success
        5th call: fail
        6th call: success
        """
        self.call_count += 1

        # Success on calls 1, 4, 6 (every 3rd starting from 1, then every 2nd, then last)
        if self.call_count in [1, 4, 6]:
            assert trigger.payload is not None
            branch_name = trigger.payload.get("message", "unknown")
            return f"Response for {branch_name}"

        # Fail otherwise
        raise UiPathPendingTriggerError(
            ErrorCategory.SYSTEM, f"Trigger is still pending (call #{self.call_count})"
        )


@pytest.mark.asyncio
async def test_parallel_branches_with_multiple_interrupts_execution():
    """Test graph execution with parallel branches and multiple interrupts."""

    # Define state
    class State(TypedDict, total=False):
        branch_a_result: str | None
        branch_b_result: str | None
        branch_c_result: str | None

    # Define nodes that interrupt
    def branch_a(state: State) -> State:
        result = interrupt({"message": "Branch A needs input"})
        return {"branch_a_result": f"A completed with: {result}"}

    def branch_b(state: State) -> State:
        result = interrupt({"message": "Branch B needs input"})
        return {"branch_b_result": f"B completed with: {result}"}

    def branch_c(state: State) -> State:
        result = interrupt({"message": "Branch C needs input"})
        return {"branch_c_result": f"C completed with: {result}"}

    # Build graph with parallel branches
    graph = StateGraph(State)
    graph.add_node("branch_a", branch_a)
    graph.add_node("branch_b", branch_b)
    graph.add_node("branch_c", branch_c)

    # All branches start in parallel
    graph.add_edge(START, "branch_a")
    graph.add_edge(START, "branch_b")
    graph.add_edge(START, "branch_c")

    # All branches go to end
    graph.add_edge("branch_a", END)
    graph.add_edge("branch_b", END)
    graph.add_edge("branch_c", END)

    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()

    try:
        # Compile graph with checkpointer
        async with AsyncSqliteSaver.from_conn_string(temp_db.name) as memory:
            compiled_graph = graph.compile(checkpointer=memory)

            # Create base runtime
            base_runtime = UiPathLangGraphRuntime(
                graph=compiled_graph,
                runtime_id="parallel-test",
                entrypoint="test",
            )

            # Create storage and trigger manager
            storage = SqliteResumableStorage(memory)

            # Wrap with UiPathResumableRuntime
            runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=MockTriggerHandler(),
                runtime_id="parallel-test",
            )

            # First execution - should hit all 3 interrupts
            result = await runtime.execute(
                input={
                    "branch_a_result": None,
                    "branch_b_result": None,
                    "branch_c_result": None,
                },
                options=UiPathExecuteOptions(resume=False),
            )

            # Should be suspended with 3 triggers
            assert result.status == UiPathRuntimeStatus.SUSPENDED
            assert result.triggers is not None
            assert len(result.triggers) == 3

            # Verify triggers were saved to storage
            saved_triggers = await storage.get_triggers("parallel-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 3

            # Resume 1: Resolve only first interrupt (no input, will restore from storage)
            result_1 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should still be suspended with 2 remaining interrupts
            assert result_1.status == UiPathRuntimeStatus.SUSPENDED
            assert result_1.triggers is not None
            assert len(result_1.triggers) == 2

            # Verify only 2 triggers remain in storage
            saved_triggers = await storage.get_triggers("parallel-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 2

            # Resume 2: Resolve second interrupt
            result_2 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should still be suspended with 1 remaining interrupt
            assert result_2.status == UiPathRuntimeStatus.SUSPENDED
            assert result_2.triggers is not None
            assert len(result_2.triggers) == 1

            # Verify only 1 trigger remains in storage
            saved_triggers = await storage.get_triggers("parallel-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 1

            # Resume 3: Resolve final interrupt
            result_3 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should now be successful
            assert result_3.status == UiPathRuntimeStatus.SUCCESSFUL
            assert result_3.output is not None

            # Verify no triggers remain
            saved_triggers = await storage.get_triggers("parallel-test")
            assert saved_triggers is None or len(saved_triggers) == 0

            # Verify all branches completed
            output = result_3.output
            assert "branch_a_result" in output
            assert "branch_b_result" in output
            assert "branch_c_result" in output

    finally:
        if os.path.exists(temp_db.name):
            os.remove(temp_db.name)
