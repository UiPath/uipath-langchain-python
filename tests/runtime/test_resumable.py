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


class SequentialTriggerHandler:
    """Mock implementation that fires triggers sequentially.

    Resolves triggers one at a time across multiple resume calls.
    """

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


class ParallelTriggerHandler:
    """Mock implementation that fires all triggers immediately.

    Resolves all triggers on the first resume call.
    """

    async def create_trigger(self, suspend_value: Any) -> UiPathResumeTrigger:
        """Create a trigger from suspend value."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            payload=suspend_value,
        )
        return trigger

    async def read_trigger(self, trigger: UiPathResumeTrigger) -> Any:
        """Read trigger and return immediate response."""
        assert trigger.payload is not None
        branch_name = trigger.payload.get("message", "unknown")
        return f"Response for {branch_name}"


@pytest.mark.asyncio
async def test_parallel_branches_with_sequential_trigger_resolution():
    """Test graph execution with parallel branches where triggers resolve sequentially."""

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
                runtime_id="parallel-sequential-test",
                entrypoint="test",
            )

            # Create storage and trigger manager
            storage = SqliteResumableStorage(memory)

            # Wrap with UiPathResumableRuntime using sequential trigger handler
            runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=SequentialTriggerHandler(),
                runtime_id="parallel-sequential-test",
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
            saved_triggers = await storage.get_triggers("parallel-sequential-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 3

            # Resume 1: Resolve only first interrupt
            result_1 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should still be suspended with 2 remaining interrupts
            assert result_1.status == UiPathRuntimeStatus.SUSPENDED
            assert result_1.triggers is not None
            assert len(result_1.triggers) == 2

            # Verify only 2 triggers remain in storage
            saved_triggers = await storage.get_triggers("parallel-sequential-test")
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
            saved_triggers = await storage.get_triggers("parallel-sequential-test")
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
            saved_triggers = await storage.get_triggers("parallel-sequential-test")
            assert saved_triggers is None or len(saved_triggers) == 0

            # Verify all branches completed
            output = result_3.output
            assert "branch_a_result" in output
            assert "branch_b_result" in output
            assert "branch_c_result" in output

    finally:
        if os.path.exists(temp_db.name):
            os.remove(temp_db.name)


@pytest.mark.asyncio
async def test_parallel_branches_with_parallel_trigger_resolution():
    """Test graph execution with parallel branches where all triggers fire immediately."""

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
                runtime_id="parallel-parallel-test",
                entrypoint="test",
            )

            # Create storage and trigger manager
            storage = SqliteResumableStorage(memory)

            # Wrap with UiPathResumableRuntime using parallel trigger handler
            runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=ParallelTriggerHandler(),
                runtime_id="parallel-parallel-test",
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
            saved_triggers = await storage.get_triggers("parallel-parallel-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 3

            # Resume: All triggers should resolve immediately
            result_resume = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should now be successful (all triggers resolved in one go)
            assert result_resume.status == UiPathRuntimeStatus.SUCCESSFUL
            assert result_resume.output is not None

            # Verify no triggers remain
            saved_triggers = await storage.get_triggers("parallel-parallel-test")
            assert saved_triggers is None or len(saved_triggers) == 0

            # Verify all branches completed
            output = result_resume.output
            assert isinstance(output, dict)
            assert "branch_a_result" in output
            assert "branch_b_result" in output
            assert "branch_c_result" in output

            # Verify all branches got their responses
            assert "Response for Branch A needs input" in output["branch_a_result"]
            assert "Response for Branch B needs input" in output["branch_b_result"]
            assert "Response for Branch C needs input" in output["branch_c_result"]

    finally:
        if os.path.exists(temp_db.name):
            os.remove(temp_db.name)


@pytest.mark.asyncio
async def test_two_branches_with_two_sequential_interrupts_each():
    """Test graph execution with 2 parallel branches, each having 2 sequential interrupts."""

    # Define state
    class State(TypedDict, total=False):
        branch_a_first_result: str | None
        branch_a_second_result: str | None
        branch_b_first_result: str | None
        branch_b_second_result: str | None

    # Define nodes that interrupt twice sequentially
    def branch_a_first(state: State) -> State:
        result = interrupt({"message": "Branch A - First interrupt"})
        return {"branch_a_first_result": f"A-1 completed with: {result}"}

    def branch_a_second(state: State) -> State:
        result = interrupt({"message": "Branch A - Second interrupt"})
        return {"branch_a_second_result": f"A-2 completed with: {result}"}

    def branch_b_first(state: State) -> State:
        result = interrupt({"message": "Branch B - First interrupt"})
        return {"branch_b_first_result": f"B-1 completed with: {result}"}

    def branch_b_second(state: State) -> State:
        result = interrupt({"message": "Branch B - Second interrupt"})
        return {"branch_b_second_result": f"B-2 completed with: {result}"}

    # Build graph with parallel branches, each with sequential nodes
    graph = StateGraph(State)
    graph.add_node("branch_a_first", branch_a_first)
    graph.add_node("branch_a_second", branch_a_second)
    graph.add_node("branch_b_first", branch_b_first)
    graph.add_node("branch_b_second", branch_b_second)

    # Branch A: START -> a_first -> a_second -> END
    graph.add_edge(START, "branch_a_first")
    graph.add_edge("branch_a_first", "branch_a_second")
    graph.add_edge("branch_a_second", END)

    # Branch B: START -> b_first -> b_second -> END
    graph.add_edge(START, "branch_b_first")
    graph.add_edge("branch_b_first", "branch_b_second")
    graph.add_edge("branch_b_second", END)

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
                runtime_id="two-branches-sequential-test",
                entrypoint="test",
            )

            # Create storage and trigger manager
            storage = SqliteResumableStorage(memory)

            # Wrap with UiPathResumableRuntime using parallel trigger handler
            runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=ParallelTriggerHandler(),
                runtime_id="two-branches-sequential-test",
            )

            # First execution - should hit first interrupt in both branches (2 total)
            result = await runtime.execute(
                input={
                    "branch_a_first_result": None,
                    "branch_a_second_result": None,
                    "branch_b_first_result": None,
                    "branch_b_second_result": None,
                },
                options=UiPathExecuteOptions(resume=False),
            )

            # Should be suspended with 2 triggers (first interrupt from each branch)
            assert result.status == UiPathRuntimeStatus.SUSPENDED
            assert result.triggers is not None
            assert len(result.triggers) == 2

            # Verify triggers were saved to storage
            saved_triggers = await storage.get_triggers("two-branches-sequential-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 2

            # Verify we got the first interrupts from both branches
            trigger_messages: list[str | None] = []
            for t in result.triggers:
                assert t.payload is not None
                assert isinstance(t.payload, dict)
                trigger_messages.append(t.payload.get("message"))
            assert "Branch A - First interrupt" in trigger_messages
            assert "Branch B - First interrupt" in trigger_messages

            # Resume 1: Resolve first interrupts, will hit second interrupts
            result_1 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should still be suspended with 2 triggers (second interrupt from each branch)
            assert result_1.status == UiPathRuntimeStatus.SUSPENDED
            assert result_1.triggers is not None
            assert len(result_1.triggers) == 2

            # Verify we got the second interrupts from both branches
            trigger_messages = []
            for t in result_1.triggers:
                assert t.payload is not None
                assert isinstance(t.payload, dict)
                trigger_messages.append(t.payload.get("message"))
            assert "Branch A - Second interrupt" in trigger_messages
            assert "Branch B - Second interrupt" in trigger_messages

            # Verify 2 triggers remain in storage
            saved_triggers = await storage.get_triggers("two-branches-sequential-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 2

            # Resume 2: Resolve second interrupts, should complete
            result_2 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should now be successful
            assert result_2.status == UiPathRuntimeStatus.SUCCESSFUL
            assert result_2.output is not None

            # Verify no triggers remain
            saved_triggers = await storage.get_triggers("two-branches-sequential-test")
            assert saved_triggers is None or len(saved_triggers) == 0

            # Verify all branch steps completed
            output = result_2.output
            assert isinstance(output, dict)
            assert "branch_a_first_result" in output
            assert "branch_a_second_result" in output
            assert "branch_b_first_result" in output
            assert "branch_b_second_result" in output

            # Verify all steps got their responses
            assert (
                "Response for Branch A - First interrupt"
                in output["branch_a_first_result"]
            )
            assert (
                "Response for Branch A - Second interrupt"
                in output["branch_a_second_result"]
            )
            assert (
                "Response for Branch B - First interrupt"
                in output["branch_b_first_result"]
            )
            assert (
                "Response for Branch B - Second interrupt"
                in output["branch_b_second_result"]
            )

    finally:
        if os.path.exists(temp_db.name):
            os.remove(temp_db.name)


@pytest.mark.asyncio
async def test_two_branches_with_two_interrupts_in_same_node():
    """Test graph execution with 2 parallel branches, each node having 2 sequential interrupts."""

    # Define state
    class State(TypedDict, total=False):
        branch_a_first_result: str | None
        branch_a_second_result: str | None
        branch_b_first_result: str | None
        branch_b_second_result: str | None

    # Define nodes that interrupt twice within the same node
    def branch_a(state: State) -> State:
        # First interrupt in branch A
        first_result = interrupt({"message": "Branch A - First interrupt"})

        # Second interrupt in branch A
        second_result = interrupt({"message": "Branch A - Second interrupt"})

        return {
            "branch_a_first_result": f"A-1 completed with: {first_result}",
            "branch_a_second_result": f"A-2 completed with: {second_result}",
        }

    def branch_b(state: State) -> State:
        # First interrupt in branch B
        first_result = interrupt({"message": "Branch B - First interrupt"})

        # Second interrupt in branch B
        second_result = interrupt({"message": "Branch B - Second interrupt"})

        return {
            "branch_b_first_result": f"B-1 completed with: {first_result}",
            "branch_b_second_result": f"B-2 completed with: {second_result}",
        }

    # Build graph with parallel branches
    graph = StateGraph(State)
    graph.add_node("branch_a", branch_a)
    graph.add_node("branch_b", branch_b)

    # Both branches start in parallel
    graph.add_edge(START, "branch_a")
    graph.add_edge(START, "branch_b")

    # Both branches go to end
    graph.add_edge("branch_a", END)
    graph.add_edge("branch_b", END)

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
                runtime_id="two-branches-same-node-test",
                entrypoint="test",
            )

            # Create storage and trigger manager
            storage = SqliteResumableStorage(memory)

            # Wrap with UiPathResumableRuntime using parallel trigger handler
            runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=ParallelTriggerHandler(),
                runtime_id="two-branches-same-node-test",
            )

            # First execution - should hit first interrupt in both branches (2 total)
            result = await runtime.execute(
                input={
                    "branch_a_first_result": None,
                    "branch_a_second_result": None,
                    "branch_b_first_result": None,
                    "branch_b_second_result": None,
                },
                options=UiPathExecuteOptions(resume=False),
            )

            # Should be suspended with 2 triggers (first interrupt from each branch)
            assert result.status == UiPathRuntimeStatus.SUSPENDED
            assert result.triggers is not None
            assert len(result.triggers) == 2

            # Verify triggers were saved to storage
            saved_triggers = await storage.get_triggers("two-branches-same-node-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 2

            # Verify we got the first interrupts from both branches
            trigger_messages: list[str | None] = []
            for t in result.triggers:
                assert t.payload is not None
                assert isinstance(t.payload, dict)
                trigger_messages.append(t.payload.get("message"))
            assert "Branch A - First interrupt" in trigger_messages
            assert "Branch B - First interrupt" in trigger_messages

            # Resume 1: Resolve first interrupts, will hit second interrupts in same nodes
            result_1 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should still be suspended with 2 triggers (second interrupt from each branch)
            assert result_1.status == UiPathRuntimeStatus.SUSPENDED
            assert result_1.triggers is not None
            assert len(result_1.triggers) == 2

            # Verify we got the second interrupts from both branches
            trigger_messages = []
            for t in result_1.triggers:
                assert t.payload is not None
                assert isinstance(t.payload, dict)
                trigger_messages.append(t.payload.get("message"))
            assert "Branch A - Second interrupt" in trigger_messages
            assert "Branch B - Second interrupt" in trigger_messages

            # Verify 2 triggers remain in storage
            saved_triggers = await storage.get_triggers("two-branches-same-node-test")
            assert saved_triggers is not None
            assert len(saved_triggers) == 2

            # Resume 2: Resolve second interrupts, should complete
            result_2 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should now be successful
            assert result_2.status == UiPathRuntimeStatus.SUCCESSFUL
            assert result_2.output is not None

            # Verify no triggers remain
            saved_triggers = await storage.get_triggers("two-branches-same-node-test")
            assert saved_triggers is None or len(saved_triggers) == 0

            # Verify all branch steps completed
            output = result_2.output
            assert isinstance(output, dict)
            assert "branch_a_first_result" in output
            assert "branch_a_second_result" in output
            assert "branch_b_first_result" in output
            assert "branch_b_second_result" in output

            # Verify all steps got their responses
            assert (
                "Response for Branch A - First interrupt"
                in output["branch_a_first_result"]
            )
            assert (
                "Response for Branch A - Second interrupt"
                in output["branch_a_second_result"]
            )
            assert (
                "Response for Branch B - First interrupt"
                in output["branch_b_first_result"]
            )
            assert (
                "Response for Branch B - Second interrupt"
                in output["branch_b_second_result"]
            )

    finally:
        if os.path.exists(temp_db.name):
            os.remove(temp_db.name)


class CustomSequentialTriggerHandler:
    """Mock implementation that fires triggers in a specific pattern.

    Resolves triggers according to a custom sequence:
    all calls (first branch, two interrupts): success
    1st/2nd call (second branch, single interrupt): fail
    3rd call (second branch, single interrupt): success
    """

    def __init__(self):
        self.call_count = 0
        self.branch_b_call_count = 0

    async def create_trigger(self, suspend_value: Any) -> UiPathResumeTrigger:
        """Create a trigger from suspend value."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            payload=suspend_value,
        )
        return trigger

    async def read_trigger(self, trigger: UiPathResumeTrigger) -> Any:
        """Read trigger and return mock response based on call pattern."""
        self.call_count += 1

        assert trigger.payload is not None
        branch_name = trigger.payload.get("message", "unknown")

        # Track Branch B calls separately
        if "Branch B" in branch_name:
            self.branch_b_call_count += 1

            # First 2 calls to Branch B fail
            if self.branch_b_call_count <= 2:
                raise UiPathPendingTriggerError(
                    ErrorCategory.SYSTEM,
                    f"Trigger is still pending (call #{self.call_count}, B call #{self.branch_b_call_count})",
                )

            # Third call to Branch B succeeds
            return f"Response for {branch_name}"

        # Branch A calls always succeed
        return f"Response for {branch_name}"


@pytest.mark.asyncio
async def test_two_branches_asymmetric_interrupts_with_custom_sequential_resolution():
    """Test with 2 branches: branch A has 2 interrupts in same node, branch B has 1 interrupt.

    Triggers resolve in pattern: A1 success, B fail, A2 success, B fail, B success
    """

    # Define state
    class State(TypedDict, total=False):
        branch_a_first_result: str | None
        branch_a_second_result: str | None
        branch_b_result: str | None

    # Branch A: 2 interrupts in same node
    def branch_a(state: State) -> State:
        # First interrupt in branch A
        first_result = interrupt({"message": "Branch A - First interrupt"})

        # Second interrupt in branch A
        second_result = interrupt({"message": "Branch A - Second interrupt"})

        return {
            "branch_a_first_result": f"A-1 completed with: {first_result}",
            "branch_a_second_result": f"A-2 completed with: {second_result}",
        }

    # Branch B: 1 interrupt in node
    def branch_b(state: State) -> State:
        result = interrupt({"message": "Branch B - Single interrupt"})
        return {"branch_b_result": f"B completed with: {result}"}

    # Build graph with parallel branches
    graph = StateGraph(State)
    graph.add_node("branch_a", branch_a)
    graph.add_node("branch_b", branch_b)

    # Both branches start in parallel
    graph.add_edge(START, "branch_a")
    graph.add_edge(START, "branch_b")

    # Both branches go to end
    graph.add_edge("branch_a", END)
    graph.add_edge("branch_b", END)

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
                runtime_id="asymmetric-custom-sequential-test",
                entrypoint="test",
            )

            # Create storage and trigger manager
            storage = SqliteResumableStorage(memory)

            # Wrap with UiPathResumableRuntime using custom sequential trigger handler
            runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=CustomSequentialTriggerHandler(),
                runtime_id="asymmetric-custom-sequential-test",
            )

            # First execution - should hit first interrupt in branch A and single interrupt in branch B (2 total)
            result = await runtime.execute(
                input={
                    "branch_a_first_result": None,
                    "branch_a_second_result": None,
                    "branch_b_result": None,
                },
                options=UiPathExecuteOptions(resume=False),
            )

            # Should be suspended with 2 triggers
            assert result.status == UiPathRuntimeStatus.SUSPENDED
            assert result.triggers is not None
            assert len(result.triggers) == 2

            # Verify triggers were saved to storage
            saved_triggers = await storage.get_triggers(
                "asymmetric-custom-sequential-test"
            )
            assert saved_triggers is not None
            assert len(saved_triggers) == 2

            # Verify we got interrupts from both branches
            trigger_messages: list[str | None] = []
            for t in result.triggers:
                assert t.payload is not None
                assert isinstance(t.payload, dict)
                trigger_messages.append(t.payload.get("message"))
            assert "Branch A - First interrupt" in trigger_messages
            assert "Branch B - Single interrupt" in trigger_messages

            # Resume 1: Call 1 succeeds (A1), Call 2 fails (B)
            # Branch A first interrupt resolves, branch B still pending
            result_1 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should still be suspended with 2 triggers:
            # - Branch A hit its second interrupt
            # - Branch B still has its single interrupt pending
            assert result_1.status == UiPathRuntimeStatus.SUSPENDED
            assert result_1.triggers is not None
            assert len(result_1.triggers) == 2

            # Verify triggers in storage
            saved_triggers = await storage.get_triggers(
                "asymmetric-custom-sequential-test"
            )
            assert saved_triggers is not None
            assert len(saved_triggers) == 2

            # Verify we have A's second interrupt and B's single interrupt
            trigger_messages = []
            for t in result_1.triggers:
                assert t.payload is not None
                assert isinstance(t.payload, dict)
                trigger_messages.append(t.payload.get("message"))
            assert "Branch A - Second interrupt" in trigger_messages
            assert "Branch B - Single interrupt" in trigger_messages

            # Resume 2: (A) resolves, (B) fails
            # Branch B still pending
            result_2 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should be suspended with 1 triggers (B)
            assert result_2.status == UiPathRuntimeStatus.SUSPENDED
            assert result_2.triggers is not None
            assert len(result_2.triggers) == 1

            # Verify triggers remain
            saved_triggers = await storage.get_triggers(
                "asymmetric-custom-sequential-test"
            )
            assert saved_triggers is not None
            assert len(saved_triggers) == 1

            # Resume 3: Call succeeds (B)
            result_3 = await runtime.execute(
                input=None,
                options=UiPathExecuteOptions(resume=True),
            )

            # Should now be successful
            assert result_3.status == UiPathRuntimeStatus.SUCCESSFUL
            assert result_3.output is not None

            # Verify no triggers remain
            saved_triggers = await storage.get_triggers(
                "asymmetric-custom-sequential-test"
            )
            assert saved_triggers is None or len(saved_triggers) == 0

            # Verify all branches completed
            output = result_3.output
            assert isinstance(output, dict)
            assert "branch_a_first_result" in output
            assert "branch_a_second_result" in output
            assert "branch_b_result" in output

            # Verify all steps got their responses
            assert (
                "Response for Branch A - First interrupt"
                in output["branch_a_first_result"]
            )
            assert (
                "Response for Branch A - Second interrupt"
                in output["branch_a_second_result"]
            )
            assert (
                "Response for Branch B - Single interrupt" in output["branch_b_result"]
            )

    finally:
        if os.path.exists(temp_db.name):
            os.remove(temp_db.name)
