"""Tests for trace context storage."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from uipath_agents._observability.sqlite_trace_context_storage import (
    SqliteTraceContextStorage,
)
from uipath_agents._observability.trace_context_storage import TraceContextData


@pytest.fixture
def mock_storage():
    """Create a mock SqliteResumableStorage with async methods."""
    storage = MagicMock()
    storage.get_value = AsyncMock(return_value=None)
    storage.set_value = AsyncMock()
    return storage


@pytest.fixture
def trace_context_storage(mock_storage):
    """Create SqliteTraceContextStorage with mock."""
    return SqliteTraceContextStorage(mock_storage)


@pytest.fixture
def sample_context():
    """Create sample trace context data."""
    return TraceContextData(
        trace_id="abc123def456789012345678901234567890",
        span_id="1234567890123456",
        parent_span_id=None,
        name="Agent run - test-agent",
        start_time="2024-01-15T10:30:00Z",
        attributes={"agentId": "test-agent", "type": "agentRun"},
        pending_tool_span_id=None,
        pending_process_span_id=None,
        pending_tool_name=None,
    )


class TestSqliteTraceContextStorage:
    """Tests for SqliteTraceContextStorage."""

    @pytest.mark.asyncio
    async def test_save_trace_context(
        self, trace_context_storage, mock_storage, sample_context
    ):
        """Test saving trace context calls storage with correct parameters."""
        await trace_context_storage.save_trace_context("runtime-123", sample_context)

        mock_storage.set_value.assert_called_once_with(
            runtime_id="runtime-123",
            namespace="trace_context",
            key="agent_span",
            value=dict(sample_context),
        )

    @pytest.mark.asyncio
    async def test_load_trace_context_returns_none_when_not_saved(
        self, trace_context_storage, mock_storage
    ):
        """Test load returns None when no context saved."""
        mock_storage.get_value = AsyncMock(return_value=None)

        result = await trace_context_storage.load_trace_context("runtime-123")

        assert result is None
        mock_storage.get_value.assert_called_once_with(
            runtime_id="runtime-123",
            namespace="trace_context",
            key="agent_span",
        )

    @pytest.mark.asyncio
    async def test_load_trace_context_returns_saved_context(
        self, trace_context_storage, mock_storage, sample_context
    ):
        """Test load returns previously saved context."""
        mock_storage.get_value = AsyncMock(return_value=dict(sample_context))

        result = await trace_context_storage.load_trace_context("runtime-123")

        assert result is not None
        assert result["trace_id"] == sample_context["trace_id"]
        assert result["span_id"] == sample_context["span_id"]
        assert result["name"] == sample_context["name"]
        assert result["attributes"] == sample_context["attributes"]

    @pytest.mark.asyncio
    async def test_clear_trace_context(self, trace_context_storage, mock_storage):
        """Test clearing trace context sets value to None."""
        await trace_context_storage.clear_trace_context("runtime-123")

        mock_storage.set_value.assert_called_once_with(
            runtime_id="runtime-123",
            namespace="trace_context",
            key="agent_span",
            value=None,
        )

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, mock_storage, sample_context):
        """Test that saved context can be loaded back correctly."""
        # Simulate storage behavior
        stored_value = None

        async def mock_set_value(**kwargs):
            nonlocal stored_value
            stored_value = kwargs["value"]

        async def mock_get_value(**kwargs):
            return stored_value

        mock_storage.set_value = AsyncMock(side_effect=mock_set_value)
        mock_storage.get_value = AsyncMock(side_effect=mock_get_value)

        storage = SqliteTraceContextStorage(mock_storage)

        # Save
        await storage.save_trace_context("runtime-123", sample_context)

        # Load
        loaded = await storage.load_trace_context("runtime-123")

        assert loaded is not None
        assert loaded["trace_id"] == sample_context["trace_id"]
        assert loaded["span_id"] == sample_context["span_id"]

        # Clear
        await storage.clear_trace_context("runtime-123")

        # Should be None now
        loaded_after_clear = await storage.load_trace_context("runtime-123")
        assert loaded_after_clear is None

    def test_uses_correct_namespace_and_key(self, trace_context_storage):
        """Test that correct namespace and key constants are used."""
        assert SqliteTraceContextStorage.NAMESPACE == "trace_context"
        assert SqliteTraceContextStorage.KEY == "agent_span"


class TestTraceContextData:
    """Tests for TraceContextData TypedDict."""

    def test_create_trace_context_data(self):
        """Test creating TraceContextData with all fields."""
        data = TraceContextData(
            trace_id="abc123",
            span_id="def456",
            parent_span_id="parent789",
            name="Test span",
            start_time="2024-01-15T10:30:00Z",
            attributes={"key": "value"},
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name=None,
        )

        assert data["trace_id"] == "abc123"
        assert data["span_id"] == "def456"
        assert data["parent_span_id"] == "parent789"
        assert data["name"] == "Test span"
        assert data["start_time"] == "2024-01-15T10:30:00Z"
        assert data["attributes"] == {"key": "value"}

    def test_trace_context_data_optional_parent(self):
        """Test TraceContextData with None parent_span_id."""
        data = TraceContextData(
            trace_id="abc123",
            span_id="def456",
            parent_span_id=None,
            name="Root span",
            start_time="2024-01-15T10:30:00Z",
            attributes={},
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name=None,
        )

        assert data["parent_span_id"] is None
