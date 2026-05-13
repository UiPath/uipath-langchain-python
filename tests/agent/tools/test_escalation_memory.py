"""Tests for escalation memory cache check and ingest."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uipath_langchain.agent.tools.escalation_memory import (
    MEMORY_CACHE_HIT_METRIC,
    MEMORY_CACHE_MISS_METRIC,
    EscalationMemorySettings,
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
    @patch("uipath_langchain.agent.tools.escalation_memory._record_custom_metric")
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_returns_cached_answer(
        self, mock_uipath_cls: MagicMock, mock_record_metric: MagicMock
    ) -> None:
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
        assert result.output == {"action": "approve", "reason": "meets criteria"}
        assert result.outcome == "approved"
        mock_record_metric.assert_called_once_with(MEMORY_CACHE_HIT_METRIC, "space-123")

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory._record_custom_metric")
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_returns_none_on_empty_results(
        self, mock_uipath_cls: MagicMock, mock_record_metric: MagicMock
    ) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_response = MagicMock()
        mock_response.results = []
        mock_sdk.memory.escalation_search_async = AsyncMock(return_value=mock_response)

        result = await _check_escalation_memory_cache("space-123", {"key": "val"})
        assert result is None
        mock_record_metric.assert_called_once_with(
            MEMORY_CACHE_MISS_METRIC, "space-123"
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_returns_none_on_failure(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.escalation_search_async = AsyncMock(
            side_effect=Exception("fail")
        )

        result = await _check_escalation_memory_cache("space-123", {"key": "val"})
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_validation_error_when_no_fields(self) -> None:
        with pytest.raises(ValueError, match="at least one configured input field"):
            await _check_escalation_memory_cache("space-123", {})

    @pytest.mark.asyncio
    async def test_raises_validation_error_when_configured_fields_do_not_match(
        self,
    ) -> None:
        settings = EscalationMemorySettings(
            fieldSettings=[{"name": "other", "weight": 1.0}]
        )

        with pytest.raises(ValueError, match="at least one configured input field"):
            await _check_escalation_memory_cache(
                "space-123",
                {"key": "val"},
                memory_settings=settings,
            )


class TestIngestEscalationMemory:
    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_calls_ingest(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.escalation_ingest_async = AsyncMock()

        await _ingest_escalation_memory(
            "space-123",
            answer='{"approved": true}',
            attributes='{"input": "test"}',
            parent_span_id="abc123",
            trace_id="def456",
            user_id="reviewer@example.com",
        )

        mock_sdk.memory.escalation_ingest_async.assert_called_once()
        request = mock_sdk.memory.escalation_ingest_async.call_args.kwargs["request"]
        assert request.span_id == "abc123"
        assert request.trace_id == "def456"
        assert request.user_id == "reviewer@example.com"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
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
            parent_span_id="abc123",
            trace_id="def456",
            user_id="reviewer@example.com",
        )
