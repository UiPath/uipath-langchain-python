"""Tests for escalation memory cache check and ingest."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict
from uipath.agent.models.agent import AgentEscalationResourceConfig
from uipath.platform.common._bindings import (
    GenericResourceOverwrite,
    _resource_overwrites,
)
from uipath.platform.memory import EscalationMemorySearchResponse

from uipath_langchain.agent.tools.escalation_memory import (
    MEMORY_CACHE_HIT_METRIC,
    MEMORY_CACHE_MISS_METRIC,
    EscalationMemoryFieldSetting,
    EscalationMemoryRetriever,
    EscalationMemorySettings,
    _build_search_fields,
    _cached_result_from_search_response,
    _check_escalation_memory_cache,
    _coerce_memory_settings,
    _get_agent_memory_space_feature,
    _get_escalation_memory_folder_path,
    _get_escalation_memory_settings,
    _get_escalation_memory_space_id,
    _get_memory_space_folder_override,
    _get_user_email,
    _get_user_id,
    _ingest_escalation_memory,
    _read_value,
    _record_custom_metric,
    _resolve_memory_space_id_by_name,
    _resolve_user_id,
    _stringify_search_value,
)

USER_GUID = "a543cbbd-f3f3-4868-bccf-f5142d2d3d7e"


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
        resource = _memory_resource(is_agent_memory_enabled=False)
        assert _get_escalation_memory_space_id(resource) is None

    def test_returns_space_id_from_extra_field(self) -> None:
        resource = _memory_resource(
            is_agent_memory_enabled=True,
            memorySpaceId="space-abc",
        )
        assert _get_escalation_memory_space_id(resource) == "space-abc"

    def test_returns_none_when_no_space_id(self) -> None:
        resource = _memory_resource(is_agent_memory_enabled=True)
        assert _get_escalation_memory_space_id(resource) is None

    def test_returns_space_id_when_escalation_memory_enabled_in_properties(
        self,
    ) -> None:
        resource = _memory_resource(
            is_agent_memory_enabled=False,
            properties={
                "memory": {
                    "isEnabled": True,
                    "memorySpaceId": "space-from-memory-properties",
                }
            },
        )

        assert (
            _get_escalation_memory_space_id(resource) == "space-from-memory-properties"
        )

    @patch("uipath_langchain.agent.tools.escalation_memory.get_execution_folder_path")
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    def test_resolves_space_id_from_agent_memory_feature(
        self,
        mock_uipath_cls: MagicMock,
        mock_get_execution_folder_path: MagicMock,
    ) -> None:
        resource = _memory_resource(
            is_agent_memory_enabled=False,
            properties={"memory": {"isEnabled": True}},
        )
        agent = SimpleNamespace(
            features=[
                {
                    "$featureType": "memorySpace",
                    "isEnabled": True,
                    "memorySpaceName": "MemorySpace",
                    "folderPath": "solution_folder",
                    "dynamicFewShotSettings": {"isEnabled": False},
                }
            ]
        )
        mock_get_execution_folder_path.return_value = "/My Workspace"
        mock_sdk = MagicMock()
        mock_sdk.memory.list.return_value = SimpleNamespace(
            value=[SimpleNamespace(id="resolved-space-id")]
        )
        mock_uipath_cls.return_value = mock_sdk

        result = _get_escalation_memory_space_id(resource, agent)

        assert result == "resolved-space-id"
        mock_sdk.memory.list.assert_called_once_with(
            filter="name eq 'MemorySpace'",
            folder_path="/My Workspace",
        )

    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    def test_resolves_agent_memory_feature_with_resource_overwrite(
        self,
        mock_uipath_cls: MagicMock,
    ) -> None:
        resource = _memory_resource(
            is_agent_memory_enabled=False,
            properties={"memory": {"isEnabled": True}},
        )
        agent = SimpleNamespace(
            features=[
                {
                    "$featureType": "memorySpace",
                    "isEnabled": True,
                    "memorySpaceName": "MemorySpace",
                    "folderPath": "solution_folder",
                    "dynamicFewShotSettings": {"isEnabled": False},
                }
            ]
        )
        mock_sdk = MagicMock()
        mock_sdk.memory.list.return_value = SimpleNamespace(
            value=[SimpleNamespace(id="resolved-space-id")]
        )
        mock_uipath_cls.return_value = mock_sdk
        token = _resource_overwrites.set(
            {
                "memorySpace.MemorySpace": GenericResourceOverwrite(
                    resource_type="memorySpace",
                    name="MemorySpace",
                    folder_path="/My Workspace/Debug_escs",
                )
            }
        )

        try:
            result = _get_escalation_memory_space_id(resource, agent)
            folder_path = _get_escalation_memory_folder_path(resource, agent)
        finally:
            _resource_overwrites.reset(token)

        assert result == "resolved-space-id"
        assert folder_path == "/My Workspace/Debug_escs"
        mock_sdk.memory.list.assert_called_once_with(
            filter="name eq 'MemorySpace'",
            folder_path="/My Workspace/Debug_escs",
        )

    def test_skips_non_memory_and_disabled_agent_features(self) -> None:
        agent = SimpleNamespace(
            features=[
                {"$featureType": "context", "memorySpaceId": "ignored-context"},
                {
                    "$featureType": "memorySpace",
                    "isEnabled": False,
                    "memorySpaceId": "ignored-disabled",
                },
                {
                    "$featureType": "memorySpace",
                    "isEnabled": True,
                    "memorySpaceId": "selected-space",
                },
            ]
        )

        feature = _get_agent_memory_space_feature(agent)

        assert feature is not None
        assert _read_value(feature, "memorySpaceId") == "selected-space"

    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    def test_space_name_resolution_returns_none_when_lookup_fails(
        self,
        mock_uipath_cls: MagicMock,
    ) -> None:
        mock_sdk = MagicMock()
        mock_sdk.memory.list.side_effect = RuntimeError("lookup failed")
        mock_uipath_cls.return_value = mock_sdk

        result = _resolve_memory_space_id_by_name("MemorySpace", "/Memory")

        assert result is None

    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    def test_space_name_resolution_returns_none_when_lookup_is_empty(
        self,
        mock_uipath_cls: MagicMock,
    ) -> None:
        mock_sdk = MagicMock()
        mock_sdk.memory.list.return_value = SimpleNamespace(value=[])
        mock_uipath_cls.return_value = mock_sdk

        result = _resolve_memory_space_id_by_name("MemorySpace", "/Memory")

        assert result is None


class TestGetMemorySpaceFolderOverride:
    def test_returns_none_without_matching_override(self) -> None:
        token = _resource_overwrites.set(
            {
                "memorySpace.OtherSpace": GenericResourceOverwrite(
                    resource_type="memorySpace",
                    name="OtherSpace",
                    folder_path="/Other",
                )
            }
        )

        try:
            result = _get_memory_space_folder_override("MemorySpace")
        finally:
            _resource_overwrites.reset(token)

        assert result is None

    def test_returns_none_when_matching_override_has_no_folder(self) -> None:
        token = _resource_overwrites.set(
            {
                "memorySpace.MemorySpace": GenericResourceOverwrite(
                    resource_type="memorySpace",
                    name="MemorySpace",
                    folder_path="",
                )
            }
        )

        try:
            result = _get_memory_space_folder_override("MemorySpace")
        finally:
            _resource_overwrites.reset(token)

        assert result is None


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

    def test_returns_settings_when_escalation_memory_enabled_in_properties(
        self,
    ) -> None:
        resource = _memory_resource(
            is_agent_memory_enabled=False,
            properties={
                "memory": {
                    "isEnabled": True,
                    "threshold": 0.8,
                    "searchMode": "hybrid",
                    "fieldSettings": [{"name": "request_details", "weight": 1}],
                }
            },
        )

        settings = _get_escalation_memory_settings(resource)

        assert settings is not None
        assert settings.threshold == 0.8
        assert settings.search_mode.value == "Hybrid"
        assert settings.field_settings == [
            EscalationMemoryFieldSetting(name="request_details", weight=1)
        ]

    def test_normalizes_semantic_search_mode(self) -> None:
        settings = EscalationMemorySettings(searchMode="semantic")

        assert settings.search_mode.value == "Semantic"


class TestGetUserEmail:
    def test_extracts_email_from_supported_shapes(self) -> None:
        assert _get_user_email(None) is None
        assert (
            _get_user_email({"emailAddress": "dict@example.com"}) == "dict@example.com"
        )
        assert _get_user_email({"email": "email@example.com"}) == "email@example.com"
        assert _get_user_email({"Email": "pascal@example.com"}) == "pascal@example.com"
        assert _get_user_email({"userName": "user@example.com"}) == "user@example.com"
        assert _get_user_email({"name": "Reviewer"}) is None
        assert (
            _get_user_email(SimpleNamespace(emailAddress="object@example.com"))
            == "object@example.com"
        )


class TestGetUserId:
    def test_extracts_user_id_from_supported_shapes(self) -> None:
        assert _get_user_id(None) is None
        assert _get_user_id({"identifier": USER_GUID}) == USER_GUID
        assert (
            _get_user_id({"identifier": "aad|cef1337c-3456-4ae9-81c9-30d033dc2bef"})
            is None
        )
        assert _get_user_id({"id": "dict-id"}) is None
        assert _get_user_id({"id": 4753819}) is None
        assert _get_user_id({"id": "4753819"}) is None
        assert _get_user_id({"userId": USER_GUID}) == USER_GUID
        assert _get_user_id({"userGlobalId": USER_GUID.upper()}) == USER_GUID
        assert _get_user_id(SimpleNamespace(identifier=USER_GUID)) == USER_GUID


class TestResolveUserId:
    @pytest.mark.asyncio
    async def test_returns_existing_user_id_without_api_call(self) -> None:
        assert await _resolve_user_id({"identifier": USER_GUID}) == USER_GUID

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPathConfig")
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_resolves_email_to_guid_identifier(
        self,
        mock_uipath_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_config.organization_id = "org-123"
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"email": "reviewer@example.com", "identifier": USER_GUID}
        ]
        mock_sdk = MagicMock()
        mock_sdk.api_client.request_async = AsyncMock(return_value=mock_response)
        mock_uipath_cls.return_value = mock_sdk

        result = await _resolve_user_id(
            {"emailAddress": "reviewer@example.com", "id": 4753819}
        )

        assert result == USER_GUID
        mock_sdk.api_client.request_async.assert_awaited_once_with(
            "GET",
            "/identity_/api/Directory/Search/org-123",
            scoped="org",
            params={
                "startsWith": "reviewer@example.com",
                "sourceFilter": ["directoryUsers", "localUsers"],
            },
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPathConfig")
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_ignores_directory_identifier_that_is_not_guid(
        self,
        mock_uipath_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_config.organization_id = "org-123"
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "email": "reviewer@example.com",
                "identifier": "aad|cef1337c-3456-4ae9-81c9-30d033dc2bef",
            }
        ]
        mock_sdk = MagicMock()
        mock_sdk.api_client.request_async = AsyncMock(return_value=mock_response)
        mock_uipath_cls.return_value = mock_sdk

        result = await _resolve_user_id({"emailAddress": "reviewer@example.com"})

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_user_has_no_email(self) -> None:
        assert await _resolve_user_id({"displayName": "Reviewer"}) is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPathConfig")
    async def test_returns_none_when_organization_id_is_missing(
        self,
        mock_config: MagicMock,
    ) -> None:
        mock_config.organization_id = None

        result = await _resolve_user_id({"emailAddress": "reviewer@example.com"})

        assert result is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPathConfig")
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_returns_none_when_directory_lookup_fails(
        self,
        mock_uipath_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_config.organization_id = "org-123"
        mock_sdk = MagicMock()
        mock_sdk.api_client.request_async = AsyncMock(
            side_effect=RuntimeError("directory unavailable")
        )
        mock_uipath_cls.return_value = mock_sdk

        result = await _resolve_user_id({"emailAddress": "reviewer@example.com"})

        assert result is None


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
    async def test_returns_cached_answer_when_sdk_response_has_string_answer(
        self, mock_uipath_cls: MagicMock, mock_record_metric: MagicMock
    ) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        validation_error: Exception | None = None
        try:
            EscalationMemorySearchResponse.model_validate(
                {
                    "results": [
                        {
                            "answer": '{"output": {"approved": true}, "outcome": "approved"}'
                        }
                    ]
                }
            )
        except Exception as error:
            validation_error = error
        assert validation_error is not None
        mock_sdk.memory.escalation_search_async = AsyncMock(
            side_effect=validation_error
        )
        mock_sdk.memory._escalation_search_spec.return_value = SimpleNamespace(
            method="POST",
            endpoint="/llmopstenant_/api/Agent/memory/space-123/escalation/search",
            headers={"x-uipath-folderkey": "folder-key"},
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"answer": '{"output": {"approved": true}, "outcome": "approved"}'}
            ]
        }
        mock_sdk.memory.request_async = AsyncMock(return_value=mock_response)

        result = await _check_escalation_memory_cache(
            "space-123",
            {"Content": "Is the sky blue?"},
            folder_path="/Memory/Folder",
        )

        assert result is not None
        assert result.output == {"approved": True}
        assert result.outcome == "approved"
        mock_sdk.memory.request_async.assert_awaited_once()
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
    @patch("uipath_langchain.agent.tools.escalation_memory._record_custom_metric")
    async def test_treats_empty_search_fields_as_cache_miss(
        self, mock_record_metric: MagicMock
    ) -> None:
        result = await _check_escalation_memory_cache("space-123", {})

        assert result is None
        mock_record_metric.assert_called_once_with(
            MEMORY_CACHE_MISS_METRIC, "space-123"
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory._record_custom_metric")
    async def test_treats_unmatched_configured_fields_as_cache_miss(
        self, mock_record_metric: MagicMock
    ) -> None:
        settings = EscalationMemorySettings(
            fieldSettings=[{"name": "other", "weight": 1.0}]
        )

        result = await _check_escalation_memory_cache(
            "space-123",
            {"key": "val"},
            memory_settings=settings,
        )

        assert result is None
        mock_record_metric.assert_called_once_with(
            MEMORY_CACHE_MISS_METRIC, "space-123"
        )

    def test_search_response_without_answer_is_cache_miss(self) -> None:
        result = _cached_result_from_search_response({"results": [{}]})

        assert result is None

    def test_search_response_with_invalid_json_answer_is_cache_miss(self) -> None:
        result = _cached_result_from_search_response(
            {"results": [{"answer": "not-json"}]}
        )

        assert result is None

    def test_search_response_without_output_is_cache_miss(self) -> None:
        result = _cached_result_from_search_response(
            {"results": [{"answer": {"outcome": "approved"}}]}
        )

        assert result is None


class TestBuildSearchFields:
    def test_search_request_includes_required_definition_prompt(self) -> None:
        request = EscalationMemoryRetriever("space-123")._build_search_request(
            {"keep": "value"}
        )

        assert request.definition_system_prompt == ""
        assert (
            request.model_dump(by_alias=True, exclude_none=True)[
                "definitionSystemPrompt"
            ]
            == ""
        )

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
            user_id=USER_GUID,
        )

        mock_sdk.memory.escalation_ingest_async.assert_called_once()
        request = mock_sdk.memory.escalation_ingest_async.call_args.kwargs["request"]
        assert request.span_id == "abc123"
        assert request.trace_id == "def456"
        assert request.user_id == USER_GUID

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_calls_ingest_without_user_id(
        self, mock_uipath_cls: MagicMock
    ) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.escalation_ingest_async = AsyncMock()

        await _ingest_escalation_memory(
            "space-123",
            answer='{"approved": true}',
            attributes='{"input": "test"}',
            parent_span_id="abc123",
            trace_id="def456",
        )

        request = mock_sdk.memory.escalation_ingest_async.call_args.kwargs["request"]
        assert request.user_id is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_memory.UiPath")
    async def test_calls_ingest_without_invalid_user_id(
        self, mock_uipath_cls: MagicMock
    ) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.escalation_ingest_async = AsyncMock()

        await _ingest_escalation_memory(
            "space-123",
            answer='{"approved": true}',
            attributes='{"input": "test"}',
            parent_span_id="abc123",
            trace_id="def456",
            user_id="aad|cef1337c-3456-4ae9-81c9-30d033dc2bef",
        )

        request = mock_sdk.memory.escalation_ingest_async.call_args.kwargs["request"]
        assert request.user_id is None

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
