"""Tests for escalation memory cache check and ingest."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict
from uipath.agent.models.agent import AgentEscalationResourceConfig

from uipath_langchain.agent.tools.escalation_memory import (
    MEMORY_CACHE_HIT_METRIC,
    MEMORY_CACHE_MISS_METRIC,
    EscalationMemoryFieldSetting,
    EscalationMemorySettings,
    _build_search_fields,
    _check_escalation_memory_cache,
    _coerce_memory_settings,
    _get_escalation_memory_settings,
    _get_escalation_memory_space_id,
    _get_user_email,
    _ingest_escalation_memory,
    _read_value,
    _record_custom_metric,
    _stringify_search_value,
)


def _memory_resource(**overrides: object) -> AgentEscalationResourceConfig:
    values: dict[str, object] = {
        "name": "approval",
        "description": "Request approval",
        "channels": [],
    }
    values.update(overrides)
    return AgentEscalationResourceConfig(**values)


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


class TestGetEscalationMemorySettings:
    def test_returns_none_when_disabled(self) -> None:
        resource = _memory_resource(is_agent_memory_enabled=False)
        assert _get_escalation_memory_settings(resource) is None

    def test_returns_none_when_memory_properties_missing(self) -> None:
        resource = _memory_resource(is_agent_memory_enabled=True, properties={})
        assert _get_escalation_memory_settings(resource) is None

    def test_returns_typed_settings_from_properties(self) -> None:
        resource = _memory_resource(
            is_agent_memory_enabled=True,
            properties={
                "memory": {
                    "threshold": 0.7,
                    "searchMode": "Semantic",
                    "fieldSettings": [{"name": "question", "weight": 0.4}],
                }
            },
        )

        settings = _get_escalation_memory_settings(resource)

        assert settings is not None
        assert settings.threshold == 0.7
        assert settings.search_mode.value == "Semantic"
        assert settings.field_settings == [
            EscalationMemoryFieldSetting(name="question", weight=0.4)
        ]


class TestGetUserEmail:
    def test_extracts_email_from_supported_shapes(self) -> None:
        assert _get_user_email(None) is None
        assert (
            _get_user_email({"emailAddress": "dict@example.com"}) == "dict@example.com"
        )
        assert _get_user_email({"name": "Reviewer"}) is None
        assert (
            _get_user_email(SimpleNamespace(emailAddress="object@example.com"))
            == "object@example.com"
        )


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


class TestBuildSearchFields:
    def test_filters_empty_and_unconfigured_fields(self) -> None:
        settings = EscalationMemorySettings(
            fieldSettings=[
                {"name": "keep", "weight": 0.25},
                {"name": "empty", "weight": 1.0},
            ]
        )

        fields = _build_search_fields(
            {
                "keep": {"answer": True},
                "empty": None,
                "ignored": "value",
            },
            settings,
        )

        assert len(fields) == 1
        assert fields[0].key_path == ["escalation-input", "keep"]
        assert fields[0].value == '{"answer": true}'
        assert fields[0].settings.weight == 0.25


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


class TestEscalationMemoryUtilities:
    def test_record_custom_metric_creates_and_reuses_counter(self, monkeypatch) -> None:
        from opentelemetry import metrics, trace

        counters: list[tuple[str, int, dict[str, str]]] = []
        events: list[tuple[str, dict[str, object]]] = []

        class Counter:
            def __init__(self, name: str) -> None:
                self.name = name

            def add(self, value: int, attributes: dict[str, str]) -> None:
                counters.append((self.name, value, attributes))

        class Meter:
            def __init__(self) -> None:
                self.created: list[str] = []

            def create_counter(self, name: str) -> Counter:
                self.created.append(name)
                return Counter(name)

        class Span:
            def is_recording(self) -> bool:
                return True

            def add_event(self, name: str, attributes: dict[str, object]) -> None:
                events.append((name, attributes))

        meter = Meter()
        monkeypatch.setattr(metrics, "get_meter", lambda _name: meter)
        monkeypatch.setattr(trace, "get_current_span", lambda: Span())

        from uipath_langchain.agent.tools import escalation_memory

        escalation_memory._metric_counters.clear()
        _record_custom_metric(MEMORY_CACHE_HIT_METRIC, "space-123")
        _record_custom_metric(MEMORY_CACHE_HIT_METRIC, "space-123")

        assert meter.created == [MEMORY_CACHE_HIT_METRIC]
        assert counters == [
            (MEMORY_CACHE_HIT_METRIC, 1, {"memorySpaceId": "space-123"}),
            (MEMORY_CACHE_HIT_METRIC, 1, {"memorySpaceId": "space-123"}),
        ]
        assert events == [
            (
                "customMetric",
                {
                    "name": MEMORY_CACHE_HIT_METRIC,
                    "value": 1,
                    "memorySpaceId": "space-123",
                },
            ),
            (
                "customMetric",
                {
                    "name": MEMORY_CACHE_HIT_METRIC,
                    "value": 1,
                    "memorySpaceId": "space-123",
                },
            ),
        ]

    def test_record_custom_metric_is_best_effort(self, monkeypatch) -> None:
        from opentelemetry import metrics

        monkeypatch.setattr(
            metrics,
            "get_meter",
            MagicMock(side_effect=RuntimeError("metrics unavailable")),
        )

        from uipath_langchain.agent.tools import escalation_memory

        escalation_memory._metric_counters.clear()
        _record_custom_metric(MEMORY_CACHE_MISS_METRIC, "space-123")

    def test_coerce_memory_settings_from_supported_shapes(self) -> None:
        class MemoryModel(BaseModel):
            threshold: float = 0.6
            searchMode: str = "Semantic"
            fieldSettings: list[dict[str, object]] = [
                {"name": "model-field", "weight": 0.5}
            ]

        class MemoryObject:
            threshold = 0.8
            searchMode = "Hybrid"
            fieldSettings = [{"name": "object-field", "weight": 0.9}]

        existing = EscalationMemorySettings(threshold=0.1)
        assert _coerce_memory_settings(existing) is existing
        assert _coerce_memory_settings(MemoryModel()).field_settings == [
            EscalationMemoryFieldSetting(name="model-field", weight=0.5)
        ]
        object_settings = _coerce_memory_settings(MemoryObject())
        assert object_settings.threshold == 0.8
        assert object_settings.field_settings == [
            EscalationMemoryFieldSetting(name="object-field", weight=0.9)
        ]

    def test_read_value_from_supported_shapes(self) -> None:
        class ExtraModel(BaseModel):
            model_config = ConfigDict(extra="allow")

        assert _read_value(None, "missing") is None
        assert _read_value({"present": "yes"}, "present") == "yes"
        assert _read_value({"other": "yes"}, "missing") is None
        assert _read_value(ExtraModel(extra_value="yes"), "extra_value") == "yes"
        assert _read_value(SimpleNamespace(present="yes"), "present") == "yes"
        assert _read_value(SimpleNamespace(), "missing") is None

    def test_stringify_search_value(self) -> None:
        assert _stringify_search_value(None) == ""
        assert _stringify_search_value("text") == "text"
        assert _stringify_search_value({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
        assert _stringify_search_value(("tuple", 1)) == "('tuple', 1)"
