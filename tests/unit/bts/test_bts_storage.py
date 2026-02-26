from unittest.mock import AsyncMock

import pytest

from uipath_agents._bts.bts_storage import SqliteBtsStateStorage


@pytest.fixture
def mock_sqlite_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.set_value = AsyncMock()
    storage.get_value = AsyncMock(return_value=None)
    return storage


@pytest.fixture
def bts_storage(mock_sqlite_storage: AsyncMock) -> SqliteBtsStateStorage:
    return SqliteBtsStateStorage(mock_sqlite_storage)


@pytest.mark.asyncio
async def test_save_bts_state(
    bts_storage: SqliteBtsStateStorage,
    mock_sqlite_storage: AsyncMock,
) -> None:
    state_dict = {
        "transaction_id": "abc123",
        "transaction_created": True,
        "transaction_name": "MyAgent",
        "transaction_fingerprint": "fp1",
        "agent_operation_id": "abc123-op456",
        "agent_operation_fingerprint": "fp2",
        "parent_operation_id": None,
    }
    await bts_storage.save_bts_state("runtime-1", state_dict)
    mock_sqlite_storage.set_value.assert_called_once_with(
        runtime_id="runtime-1",
        namespace="bts_state",
        key="bts_context",
        value=state_dict,
    )


@pytest.mark.asyncio
async def test_load_bts_state_returns_none_when_empty(
    bts_storage: SqliteBtsStateStorage,
) -> None:
    result = await bts_storage.load_bts_state("runtime-1")
    assert result is None


@pytest.mark.asyncio
async def test_load_bts_state_returns_saved_dict(
    bts_storage: SqliteBtsStateStorage,
    mock_sqlite_storage: AsyncMock,
) -> None:
    saved = {
        "transaction_id": "abc123",
        "transaction_created": True,
        "transaction_name": "MyAgent",
        "transaction_fingerprint": "fp1",
        "agent_operation_id": "abc123-op456",
        "agent_operation_fingerprint": "fp2",
        "parent_operation_id": None,
    }
    mock_sqlite_storage.get_value = AsyncMock(return_value=saved)
    result = await bts_storage.load_bts_state("runtime-1")
    assert result == saved


@pytest.mark.asyncio
async def test_clear_bts_state(
    bts_storage: SqliteBtsStateStorage,
    mock_sqlite_storage: AsyncMock,
) -> None:
    await bts_storage.clear_bts_state("runtime-1")
    mock_sqlite_storage.set_value.assert_called_once_with(
        runtime_id="runtime-1",
        namespace="bts_state",
        key="bts_context",
        value=None,
    )
