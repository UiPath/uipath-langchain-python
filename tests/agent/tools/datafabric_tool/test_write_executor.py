"""Tests for Data Fabric write executor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from uipath_langchain.agent.tools.datafabric_tool.models import (
    DataFabricWriteInput,
    EntityWriteOperation,
)
from uipath_langchain.agent.tools.datafabric_tool.write_executor import WriteExecutor


def _mock_entities_service() -> MagicMock:
    """Create a mock EntitiesService with async CRUD methods."""
    svc = MagicMock()
    svc.insert_record_async = AsyncMock()
    svc.update_record_async = AsyncMock()
    svc.delete_record_async = AsyncMock()
    return svc


def _mock_entity_record(record_id: str = "rec-123") -> MagicMock:
    """Create a mock EntityRecord."""
    record = MagicMock()
    record.id = record_id
    record.model_dump.return_value = {"Id": record_id, "Name": "Test"}
    return record


class TestWriteExecutor:
    """Tests for WriteExecutor.execute."""

    @pytest.mark.asyncio
    async def test_insert_calls_insert_record_async(self) -> None:
        svc = _mock_entities_service()
        record = _mock_entity_record("rec-new")
        svc.insert_record_async.return_value = record

        executor = WriteExecutor(svc)
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"Name": "New Order"},
        )
        result = await executor.execute(intent)

        svc.insert_record_async.assert_called_once_with("Orders", {"Name": "New Order"})
        assert result.success is True
        assert result.operation == "insert"
        assert result.entity_key == "Orders"
        assert result.record_id == "rec-new"
        assert result.record is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_update_calls_update_record_async(self) -> None:
        svc = _mock_entities_service()
        record = _mock_entity_record("rec-1")
        svc.update_record_async.return_value = record

        executor = WriteExecutor(svc)
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            record_id="rec-1",
            fields={"Amount": 200},
        )
        result = await executor.execute(intent)

        svc.update_record_async.assert_called_once_with(
            "Orders", "rec-1", {"Amount": 200}
        )
        assert result.success is True
        assert result.operation == "update"
        assert result.record_id == "rec-1"

    @pytest.mark.asyncio
    async def test_delete_calls_delete_record_async(self) -> None:
        svc = _mock_entities_service()
        svc.delete_record_async.return_value = None

        executor = WriteExecutor(svc)
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.delete,
            record_id="rec-1",
        )
        result = await executor.execute(intent)

        svc.delete_record_async.assert_called_once_with("Orders", "rec-1")
        assert result.success is True
        assert result.operation == "delete"
        assert result.record_id == "rec-1"
        assert result.record is None

    @pytest.mark.asyncio
    async def test_insert_error_returns_failure(self) -> None:
        svc = _mock_entities_service()
        svc.insert_record_async.side_effect = RuntimeError("API error: 403 Forbidden")

        executor = WriteExecutor(svc)
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"Name": "Test"},
        )
        result = await executor.execute(intent)

        assert result.success is False
        assert result.operation == "insert"
        assert result.entity_key == "Orders"
        assert "403 Forbidden" in (result.error or "")

    @pytest.mark.asyncio
    async def test_update_error_returns_failure(self) -> None:
        svc = _mock_entities_service()
        svc.update_record_async.side_effect = RuntimeError("Not found")

        executor = WriteExecutor(svc)
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            record_id="rec-1",
            fields={"Amount": 100},
        )
        result = await executor.execute(intent)

        assert result.success is False
        assert "Not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_delete_error_returns_failure(self) -> None:
        svc = _mock_entities_service()
        svc.delete_record_async.side_effect = RuntimeError("Not found")

        executor = WriteExecutor(svc)
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.delete,
            record_id="rec-1",
        )
        result = await executor.execute(intent)

        assert result.success is False
        assert result.record_id == "rec-1"
        assert "Not found" in (result.error or "")
