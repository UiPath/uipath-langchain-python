"""Escalation memory support for Action Center escalation tools."""

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uipath.agent.models.agent import AgentEscalationResourceConfig
from uipath.platform import UiPath
from uipath.platform.memory import (
    EscalationMemoryIngestRequest,
    FieldSettings,
    MemorySearchRequest,
    SearchField,
    SearchMode,
    SearchSettings,
)

from uipath_langchain._utils import set_current_span_error, set_span_attribute

logger = logging.getLogger(__name__)

MEMORY_CACHE_HIT_METRIC = "MemoryCacheHit"
MEMORY_CACHE_MISS_METRIC = "MemoryCacheMiss"

_metric_counters: dict[str, Any] = {}


class EscalationMemoryFieldSetting(BaseModel):
    """Per-field search configuration for escalation memory."""

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    name: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class EscalationMemorySettings(BaseModel):
    """Search settings configured on an escalation memory resource."""

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    search_mode: SearchMode = Field(default=SearchMode.Hybrid, alias="searchMode")
    field_settings: list[EscalationMemoryFieldSetting] | None = Field(
        default=None,
        alias="fieldSettings",
    )


class EscalationMemoryCachedResult(BaseModel):
    """Cached escalation output returned by memory search."""

    output: Any = None
    outcome: str | None = None


class EscalationMemoryRetriever:
    """Retrieves previously resolved escalation outcomes from UiPath memory."""

    def __init__(
        self,
        memory_space_id: str,
        *,
        folder_path: str | None = None,
        memory_settings: EscalationMemorySettings | None = None,
        uipath_sdk: UiPath | None = None,
    ) -> None:
        self.memory_space_id = memory_space_id
        self.folder_path = folder_path
        self.memory_settings = memory_settings or EscalationMemorySettings()
        self._uipath_sdk = uipath_sdk

    async def aretrieve(
        self,
        serialized_input: dict[str, Any],
    ) -> EscalationMemoryCachedResult | None:
        """Search escalation memory and return the first cached answer."""
        request = self._build_search_request(serialized_input)
        sdk = self._uipath_sdk if self._uipath_sdk is not None else UiPath()
        response = await sdk.memory.escalation_search_async(
            memory_space_id=self.memory_space_id,
            request=request,
            folder_path=self.folder_path,
        )

        results = response.results or []
        if not results or not results[0].answer:
            return None

        answer = results[0].answer
        return EscalationMemoryCachedResult(
            output=answer.output,
            outcome=answer.outcome,
        )

    def _build_search_request(
        self,
        serialized_input: dict[str, Any],
    ) -> MemorySearchRequest:
        fields = _build_search_fields(serialized_input, self.memory_settings)
        return MemorySearchRequest(
            fields=fields,
            settings=SearchSettings(
                threshold=self.memory_settings.threshold,
                result_count=1,
                search_mode=self.memory_settings.search_mode,
            ),
        )


def _get_escalation_memory_space_id(
    resource: AgentEscalationResourceConfig,
) -> str | None:
    """Resolve memory space ID from escalation resource extra fields."""
    if not resource.is_agent_memory_enabled:
        return None

    memory_space_id = _read_value(resource, "memorySpaceId", "memory_space_id")
    return str(memory_space_id) if memory_space_id else None


def _get_escalation_memory_settings(
    resource: AgentEscalationResourceConfig,
) -> EscalationMemorySettings | None:
    """Extract memory settings from escalation resource properties."""
    if not resource.is_agent_memory_enabled:
        return None

    properties = _read_value(resource, "properties")
    memory = _read_value(properties, "memory") if properties is not None else None
    if memory is None:
        return None
    return _coerce_memory_settings(memory)


def _get_user_email(user: Any) -> str | None:
    """Extract an email address from an Action Center user payload."""
    if user is None:
        return None
    if isinstance(user, dict):
        return user.get("emailAddress")
    return getattr(user, "emailAddress", None)


async def _check_escalation_memory_cache(
    memory_space_id: str,
    serialized_input: dict[str, Any],
    folder_path: str | None = None,
    memory_settings: EscalationMemorySettings | None = None,
) -> EscalationMemoryCachedResult | None:
    """Check escalation memory for a cached answer."""
    retriever = EscalationMemoryRetriever(
        memory_space_id,
        folder_path=folder_path,
        memory_settings=memory_settings,
    )

    try:
        cached_result = await retriever.aretrieve(serialized_input)
    except ValueError:
        raise
    except Exception as error:
        set_current_span_error(error)
        logger.warning(
            "Escalation memory search failed for space '%s'",
            memory_space_id,
            exc_info=True,
        )
        return None

    if cached_result is None:
        _record_custom_metric(MEMORY_CACHE_MISS_METRIC, memory_space_id)
        return None

    _record_custom_metric(MEMORY_CACHE_HIT_METRIC, memory_space_id)
    logger.info("Escalation memory cache hit for space '%s'", memory_space_id)
    set_span_attribute("fromMemory", True)
    return cached_result


async def _ingest_escalation_memory(
    memory_space_id: str,
    answer: str,
    attributes: str,
    parent_span_id: str,
    trace_id: str,
    user_id: str,
    folder_path: str | None = None,
) -> None:
    """Persist a resolved escalation outcome into memory."""
    set_span_attribute("fromMemory", False)

    try:
        request = EscalationMemoryIngestRequest(
            span_id=parent_span_id,
            trace_id=trace_id,
            answer=answer,
            attributes=attributes,
            user_id=user_id,
        )
        sdk = UiPath()
        await sdk.memory.escalation_ingest_async(
            memory_space_id=memory_space_id,
            request=request,
            folder_path=folder_path,
        )
        set_span_attribute("savedToMemory", True)
        logger.info(
            "Ingested escalation outcome into memory space '%s'", memory_space_id
        )
    except Exception as error:
        set_span_attribute("savedToMemory", False)
        set_current_span_error(error)
        logger.warning(
            "Failed to ingest escalation outcome into memory space '%s'",
            memory_space_id,
            exc_info=True,
        )


def _build_search_fields(
    serialized_input: dict[str, Any],
    memory_settings: EscalationMemorySettings,
) -> list[SearchField]:
    field_settings = memory_settings.field_settings
    field_settings_lookup = (
        {field_setting.name: field_setting for field_setting in field_settings}
        if field_settings is not None
        else None
    )

    fields: list[SearchField] = []
    for name, value in serialized_input.items():
        value_str = _stringify_search_value(value)
        if not value_str:
            continue
        if field_settings_lookup is not None and name not in field_settings_lookup:
            continue
        settings = FieldSettings()
        if field_settings_lookup is not None:
            settings = FieldSettings(weight=field_settings_lookup[name].weight)
        fields.append(
            SearchField(
                key_path=["escalation-input", name],
                value=value_str,
                settings=settings,
            )
        )

    if not fields:
        raise ValueError(
            "Escalation memory search requires at least one configured input field."
        )
    return fields


def _record_custom_metric(metric_name: str, memory_space_id: str) -> None:
    attributes = {"memorySpaceId": memory_space_id}
    try:
        from opentelemetry import metrics, trace

        counter = _metric_counters.get(metric_name)
        if counter is None:
            counter = metrics.get_meter(
                "uipath_langchain.escalation_memory"
            ).create_counter(metric_name)
            _metric_counters[metric_name] = counter
        counter.add(1, attributes)

        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(
                "customMetric",
                {
                    "name": metric_name,
                    "value": 1,
                    **attributes,
                },
            )
    except Exception:
        logger.debug("Failed to record metric '%s'", metric_name, exc_info=True)


def _coerce_memory_settings(memory: Any) -> EscalationMemorySettings:
    if isinstance(memory, EscalationMemorySettings):
        return memory
    if isinstance(memory, BaseModel):
        memory = memory.model_dump(by_alias=True, exclude_none=True)
    elif not isinstance(memory, dict):
        memory = {
            key: getattr(memory, key)
            for key in (
                "threshold",
                "searchMode",
                "search_mode",
                "fieldSettings",
                "field_settings",
            )
            if hasattr(memory, key)
        }
    return EscalationMemorySettings.model_validate(memory)


def _read_value(source: Any, *keys: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        for key in keys:
            if key in source:
                return source[key]
        return None
    if isinstance(source, BaseModel):
        extra = source.model_extra or {}
        for key in keys:
            if key in extra:
                return extra[key]
    for key in keys:
        try:
            return getattr(source, key)
        except AttributeError:
            continue
    return None


def _stringify_search_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float | list | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)
