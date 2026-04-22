"""Tests for escalation memory cache check and ingest."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uipath_langchain.agent.tools.escalation_tool import (
    EscalationAction,
    _check_escalation_memory_cache,
    _get_escalation_memory_space_id,
    _ingest_escalation_memory,
)


class TestGetEscalationMemorySpaceId:
    def test_returns_none_when_disabled(self) -> None:
        resource = MagicMock()
        resource.is_agent_memory_enabled = False
        assert _get_escalation_memory_space_id(resource) is None

    def test_returns_space_id_from_extra_field(self) -> None:
        resource = MagicMock()
        resource.is_agent_memory_enabled = True
        resource.memorySpaceId = "space-abc"
        assert _get_escalation_memory_space_id(resource) == "space-abc"

    def test_returns_none_when_no_space_id(self) -> None:
        resource = MagicMock()
        resource.is_agent_memory_enabled = True
        del resource.memorySpaceId
        del resource.memory_space_id
        assert _get_escalation_memory_space_id(resource) is None


class TestCheckEscalationMemoryCache:
    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_returns_cached_answer(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk

        cached_answer = MagicMock()
        cached_answer.output = {"action": "approve", "reason": "meets criteria"}
        cached_answer.outcome = "approved"

        mock_match = MagicMock()
        mock_match.answer = cached_answer

        mock_response = MagicMock()
        mock_response.results = [mock_match]
        mock_sdk.memory.escalation_search_async = AsyncMock(return_value=mock_response)

        result = await _check_escalation_memory_cache(
            "space-123", {"Content": "Is the sky blue?"}
        )

        assert result is not None
        assert result["action"] == EscalationAction.CONTINUE
        assert result["output"] == {"action": "approve", "reason": "meets criteria"}
        assert result["outcome"] == "approved"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_returns_none_on_empty_results(
        self, mock_uipath_cls: MagicMock
    ) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_response = MagicMock()
        mock_response.results = []
        mock_sdk.memory.escalation_search_async = AsyncMock(return_value=mock_response)

        result = await _check_escalation_memory_cache("space-123", {"key": "val"})
        assert result is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_returns_none_on_failure(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.escalation_search_async = AsyncMock(
            side_effect=Exception("fail")
        )

        result = await _check_escalation_memory_cache("space-123", {"key": "val"})
        assert result is None


class TestIngestEscalationMemory:
    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_calls_ingest(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.escalation_ingest_async = AsyncMock()

        await _ingest_escalation_memory(
            "space-123",
            answer='{"approved": true}',
            attributes='{"input": "test"}',
            span_id="abc123",
            trace_id="def456",
        )

        mock_sdk.memory.escalation_ingest_async.assert_called_once()

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_graceful_on_failure(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.escalation_ingest_async = AsyncMock(
            side_effect=Exception("fail")
        )

        # Should not raise
        await _ingest_escalation_memory(
            "space-123",
            answer="yes",
            attributes="{}",
            span_id="abc123",
            trace_id="def456",
        )
