"""Tests for SqliteResumableStorage."""

import asyncio
import os
import tempfile
from typing import Any

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from uipath.runtime import (
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)

from uipath_langchain.runtime.storage import SqliteResumableStorage


@pytest.fixture
async def storage():
    """Create a SqliteResumableStorage instance with temporary database file."""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()

    try:
        async with AsyncSqliteSaver.from_conn_string(temp_db.name) as memory:
            storage = SqliteResumableStorage(memory)
            await storage._ensure_table()

            yield storage
    finally:
        if os.path.exists(temp_db.name):
            os.remove(temp_db.name)


class TestKeyValueStorage:
    """Tests for key-value storage functionality."""

    @pytest.mark.asyncio
    async def test_set_and_get_string_value(self, storage: SqliteResumableStorage):
        """Test storing and retrieving a string value."""
        await storage.set_value("runtime1", "namespace1", "key1", "test_value")
        result = await storage.get_value("runtime1", "namespace1", "key1")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_set_and_get_dict_value(self, storage: SqliteResumableStorage):
        """Test storing and retrieving a dictionary value."""
        test_dict = {"foo": "bar", "nested": {"key": "value"}}
        await storage.set_value("runtime1", "namespace1", "key1", test_dict)
        result = await storage.get_value("runtime1", "namespace1", "key1")
        assert result == test_dict

    @pytest.mark.asyncio
    async def test_set_value_overrides_existing(self, storage: SqliteResumableStorage):
        """Test that set_value overrides existing values."""
        await storage.set_value("runtime1", "namespace1", "key1", "first_value")
        await storage.set_value("runtime1", "namespace1", "key1", "second_value")
        result = await storage.get_value("runtime1", "namespace1", "key1")
        assert result == "second_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_value_returns_none(
        self, storage: SqliteResumableStorage
    ):
        """Test that getting a non-existent value returns None."""
        result = await storage.get_value("runtime1", "namespace1", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_values_scoped_by_runtime_and_namespace(
        self, storage: SqliteResumableStorage
    ):
        """Test that values are properly scoped by runtime_id and namespace."""
        await storage.set_value("runtime1", "namespace1", "key1", "value1")
        await storage.set_value("runtime2", "namespace1", "key1", "value2")
        await storage.set_value("runtime1", "namespace2", "key1", "value3")

        assert await storage.get_value("runtime1", "namespace1", "key1") == "value1"
        assert await storage.get_value("runtime2", "namespace1", "key1") == "value2"
        assert await storage.get_value("runtime1", "namespace2", "key1") == "value3"


class TestTriggerStorage:
    """Tests for trigger storage functionality."""

    @pytest.mark.asyncio
    async def test_save_and_get_single_trigger(self, storage: SqliteResumableStorage):
        """Test saving and retrieving a single trigger."""
        trigger = UiPathResumeTrigger(
            interrupt_id="interrupt1",
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            item_key="key1",
            payload="test payload",
        )

        await storage.save_triggers("runtime1", [trigger])
        triggers = await storage.get_triggers("runtime1")

        assert triggers is not None
        assert len(triggers) == 1
        assert triggers[0].interrupt_id == "interrupt1"
        assert triggers[0].trigger_type == UiPathResumeTriggerType.API
        assert triggers[0].trigger_name == UiPathResumeTriggerName.API
        assert triggers[0].item_key == "key1"
        assert triggers[0].payload == "test payload"

    @pytest.mark.asyncio
    async def test_save_triggers_overrides_previous(
        self, storage: SqliteResumableStorage
    ):
        """Test that save_triggers replaces all previous triggers."""
        # Save first set of triggers
        trigger1 = UiPathResumeTrigger(
            interrupt_id="interrupt1",
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            payload="payload1",
        )
        trigger2 = UiPathResumeTrigger(
            interrupt_id="interrupt2",
            trigger_type=UiPathResumeTriggerType.TASK,
            trigger_name=UiPathResumeTriggerName.TASK,
            payload="payload2",
        )
        await storage.save_triggers("runtime1", [trigger1, trigger2])

        # Verify first set
        triggers = await storage.get_triggers("runtime1")
        assert triggers is not None
        assert len(triggers) == 2

        # Save second set of triggers - should replace first set
        trigger3 = UiPathResumeTrigger(
            interrupt_id="interrupt3",
            trigger_type=UiPathResumeTriggerType.JOB,
            trigger_name=UiPathResumeTriggerName.JOB,
            payload="payload3",
        )
        await storage.save_triggers("runtime1", [trigger3])

        # Verify only second set exists
        triggers = await storage.get_triggers("runtime1")
        assert triggers is not None
        assert len(triggers) == 1
        assert triggers[0].interrupt_id == "interrupt3"
        assert triggers[0].trigger_type == UiPathResumeTriggerType.JOB

    @pytest.mark.asyncio
    async def test_save_empty_list_deletes_triggers(
        self, storage: SqliteResumableStorage
    ):
        """Test that saving empty list doesn't clear existing triggers."""
        trigger = UiPathResumeTrigger(
            interrupt_id="interrupt1",
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            payload="test",
        )
        await storage.save_triggers("runtime1", [trigger])

        # Save empty list
        await storage.save_triggers("runtime1", [])

        # Verify trigger no longer exists
        triggers = await storage.get_triggers("runtime1")
        assert triggers is None

    @pytest.mark.asyncio
    async def test_delete_trigger(self, storage: SqliteResumableStorage):
        """Test deleting a specific trigger."""
        trigger1 = UiPathResumeTrigger(
            interrupt_id="interrupt1",
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            payload="payload1",
        )
        trigger2 = UiPathResumeTrigger(
            interrupt_id="interrupt2",
            trigger_type=UiPathResumeTriggerType.TASK,
            trigger_name=UiPathResumeTriggerName.TASK,
            payload="payload2",
        )
        await storage.save_triggers("runtime1", [trigger1, trigger2])

        # Delete first trigger
        await storage.delete_trigger("runtime1", trigger1)

        # Verify only second trigger remains
        triggers = await storage.get_triggers("runtime1")
        assert triggers is not None
        assert len(triggers) == 1
        assert triggers[0].interrupt_id == "interrupt2"

    @pytest.mark.asyncio
    async def test_get_nonexistent_triggers_returns_none(
        self, storage: SqliteResumableStorage
    ):
        """Test that getting triggers for non-existent runtime returns None."""
        triggers = await storage.get_triggers("nonexistent_runtime")
        assert triggers is None

    @pytest.mark.asyncio
    async def test_triggers_scoped_by_runtime_id(self, storage: SqliteResumableStorage):
        """Test that triggers are properly scoped by runtime_id."""
        trigger1 = UiPathResumeTrigger(
            interrupt_id="interrupt1",
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API,
            payload="runtime1_payload",
        )
        trigger2 = UiPathResumeTrigger(
            interrupt_id="interrupt2",
            trigger_type=UiPathResumeTriggerType.TASK,
            trigger_name=UiPathResumeTriggerName.TASK,
            payload="runtime2_payload",
        )

        await storage.save_triggers("runtime1", [trigger1])
        await storage.save_triggers("runtime2", [trigger2])

        triggers1 = await storage.get_triggers("runtime1")
        triggers2 = await storage.get_triggers("runtime2")

        assert triggers1 is not None
        assert len(triggers1) == 1
        assert triggers1[0].payload == "runtime1_payload"

        assert triggers2 is not None
        assert len(triggers2) == 1
        assert triggers2[0].payload == "runtime2_payload"

    @pytest.mark.asyncio
    async def test_multiple_parallel_workflows(self, storage: SqliteResumableStorage):
        """Test handling multiple parallel workflows with concurrent operations."""
        workflows: list[tuple[str, list[UiPathResumeTrigger], dict[str, Any]]] = []
        triggers: list[UiPathResumeTrigger] | None
        # Create multiple parallel workflows
        for i in range(10):
            runtime_id = f"workflow-{i}"
            triggers = [
                UiPathResumeTrigger(
                    interrupt_id=f"interrupt-{i}",
                    trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
                    trigger_name=UiPathResumeTriggerName.QUEUE_ITEM,
                    item_key=f"queue-item-{i}",
                    payload={"step": i % 3},
                )
            ]
            context = {
                "workflow_id": str(i),
                "data": f"data-{i}",
                "step": i % 3,
            }
            workflows.append((runtime_id, triggers, context))

        # Save all workflows concurrently
        save_tasks = []
        for runtime_id, triggers, context in workflows:
            save_tasks.append(storage.save_triggers(runtime_id, triggers))
            save_tasks.append(
                storage.set_value(
                    runtime_id, "meta", "workflow_id", context["workflow_id"]
                )
            )
            save_tasks.append(storage.set_value(runtime_id, "meta", "status", "active"))
            save_tasks.append(
                storage.set_value(runtime_id, "data", "payload", context["data"])
            )

        await asyncio.gather(*save_tasks)

        # Verify all workflows were saved correctly
        for runtime_id, expected_triggers, expected_context in workflows:
            triggers = await storage.get_triggers(runtime_id)
            workflow_id = await storage.get_value(runtime_id, "meta", "workflow_id")
            status = await storage.get_value(runtime_id, "meta", "status")
            data = await storage.get_value(runtime_id, "data", "payload")

            assert triggers is not None
            assert len(triggers) == 1
            assert triggers[0].interrupt_id == expected_triggers[0].interrupt_id
            assert triggers[0].item_key == expected_triggers[0].item_key
            assert workflow_id == expected_context["workflow_id"]
            assert status == "active"
            assert data == expected_context["data"]

    @pytest.mark.asyncio
    async def test_kv_isolation_stress_test(self, storage: SqliteResumableStorage):
        """Stress test KV store isolation."""
        tasks = []

        # Create many KV operations with different combinations
        for runtime in range(10):
            for namespace in range(10):
                for key in range(10):
                    tasks.append(
                        storage.set_value(
                            f"runtime-{runtime}",
                            f"ns-{namespace}",
                            f"key-{key}",
                            f"value-{runtime}-{namespace}-{key}",
                        )
                    )

        await asyncio.gather(*tasks)

        # Verify each value is correctly isolated
        for runtime in range(10):
            for namespace in range(10):
                for key in range(10):
                    value = await storage.get_value(
                        f"runtime-{runtime}", f"ns-{namespace}", f"key-{key}"
                    )
                    expected = f"value-{runtime}-{namespace}-{key}"
                    assert value == expected, f"Expected {expected}, got {value}"
