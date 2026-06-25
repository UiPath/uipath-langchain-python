"""Escalation memory support for Action Center escalation tools."""

import json
import logging
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from uipath.agent.models.agent import AgentEscalationResourceConfig
from uipath.platform import UiPath
from uipath.platform.common import UiPathConfig
from uipath.platform.common._bindings import _resource_overwrites
from uipath.platform.memory import (
    EscalationMemoryIngestRequest,
    FieldSettings,
    MemorySearchRequest,
    SearchField,
    SearchMode,
    SearchSettings,
)

from uipath_langchain._utils import (
    get_execution_folder_path,
    set_current_span_error,
    set_span_attribute,
)

logger = logging.getLogger(__name__)

MEMORY_CACHE_HIT_METRIC = "MemoryCacheHit"
MEMORY_CACHE_MISS_METRIC = "MemoryCacheMiss"
ESCALATION_MEMORY_STRATEGY = "EscalationMemoryCache"

_metric_counters: dict[str, Any] = {}
_MISSING_VALUE = object()


@contextmanager
def _noop_context():
    """No-op context manager when OTel is unavailable."""
    yield None


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

    @field_validator("search_mode", mode="before")
    @classmethod
    def _normalize_search_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.lower()
            if normalized == "hybrid":
                return SearchMode.Hybrid
            if normalized == "semantic":
                return SearchMode.Semantic
        return value


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
        memory_space_name: str | None = None,
        folder_path: str | None = None,
        memory_settings: EscalationMemorySettings | None = None,
        uipath_sdk: UiPath | None = None,
    ) -> None:
        self.memory_space_id = memory_space_id
        self.memory_space_name = memory_space_name or ""
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

        results_count = 0
        cached_result: EscalationMemoryCachedResult | None = None
        try:
            # Keep the OTel import local to match episodic memory and keep this
            # module importable in runtimes where tracing is not installed.
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("uipath_langchain.memory")
        except ImportError:
            tracer = None
            otel_trace = None  # type: ignore[assignment]

        # Span attribute keys matching what the LlmOpsHttpExporter and
        # Studio UI expect. "openinference.span.kind" sets SpanType.
        lookup_span_ctx = (
            tracer.start_as_current_span(
                "Find previous memories",
                attributes={
                    "openinference.span.kind": "agentMemoryLookup",
                    "type": "agentMemoryLookup",
                    "span_type": "agentMemoryLookup",
                    "uipath.custom_instrumentation": True,
                    "memorySpaceName": self.memory_space_name,
                    "memorySpaceId": self.memory_space_id,
                    "strategy": ESCALATION_MEMORY_STRATEGY,
                },
            )
            if tracer
            else _noop_context()
        )

        with lookup_span_ctx as lookup_span:
            fewshot_span_ctx = (
                tracer.start_as_current_span(
                    "Apply escalation memory",
                    attributes={
                        # LlmOps/Studio still key memory rendering off this
                        # exported span type; rename it when that contract changes.
                        "openinference.span.kind": "applyDynamicFewShot",
                        "type": "applyDynamicFewShot",
                        "span_type": "applyDynamicFewShot",
                        "uipath.custom_instrumentation": True,
                        "memorySpaceName": self.memory_space_name,
                        "memorySpaceId": self.memory_space_id,
                        "strategy": ESCALATION_MEMORY_STRATEGY,
                    },
                )
                if tracer
                else _noop_context()
            )

            with fewshot_span_ctx as fewshot_span:
                try:
                    try:
                        response = await sdk.memory.escalation_search_async(
                            memory_space_id=self.memory_space_id,
                            request=request,
                            folder_path=self.folder_path,
                        )
                    except ValidationError:
                        # Some existing escalation memories store `answer` as a
                        # JSON string that the SDK response model rejects. The
                        # raw API payload is still usable and parsed below.
                        response = await self._raw_escalation_search(sdk, request)

                    results = _read_value(response, "results") or []
                    results_count = _safe_len(results)
                    cached_result = _cached_result_from_search_response(response)
                    # Set request/response on fewshot span as JSON strings.
                    # The exporter parses JSON strings back to objects.
                    # The UI reads "response" to display matched memory items.
                    if fewshot_span and hasattr(fewshot_span, "set_attribute"):
                        fewshot_span.set_attribute(
                            "request",
                            _json_dumps(
                                request.model_dump(by_alias=True, exclude_none=True)
                            ),
                        )
                        fewshot_span.set_attribute(
                            "response",
                            _serialize_search_response_for_trace(response),
                        )
                        fewshot_span.set_attribute(
                            "fromMemory", cached_result is not None
                        )
                except Exception as error:
                    error_detail = repr(error)
                    if otel_trace:
                        if fewshot_span and hasattr(fewshot_span, "set_status"):
                            fewshot_span.set_status(
                                otel_trace.StatusCode.ERROR, error_detail
                            )
                        if lookup_span and hasattr(lookup_span, "set_status"):
                            lookup_span.set_status(
                                otel_trace.StatusCode.ERROR, error_detail
                            )
                    raise

            if lookup_span and hasattr(lookup_span, "set_attribute"):
                lookup_span.set_attribute("memoryItemsMatched", results_count)
                if cached_result is not None:
                    lookup_span.set_attribute(
                        "result",
                        _json_dumps(
                            cached_result.model_dump(by_alias=True, exclude_none=True)
                        ),
                    )

        return cached_result

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
            definition_system_prompt="",
        )

    async def _raw_escalation_search(
        self,
        sdk: UiPath,
        request: MemorySearchRequest,
    ) -> Any:
        spec = sdk.memory._escalation_search_spec(
            self.memory_space_id,
            folder_path=self.folder_path,
        )
        response = await sdk.memory.request_async(
            spec.method,
            spec.endpoint,
            json=request.model_dump(by_alias=True, exclude_none=True),
            headers=spec.headers,
        )
        return response.json()


def _get_escalation_memory_space_id(
    resource: AgentEscalationResourceConfig,
    agent: Any | None = None,
) -> str | None:
    """Resolve memory space ID from escalation resource or agent memory feature."""
    if not _is_escalation_memory_enabled(resource):
        return None

    memory = _get_escalation_memory_properties(resource)
    memory_space_id = _read_first_value(
        (resource, memory),
        "memorySpaceId",
        "memory_space_id",
    )
    if memory_space_id:
        return str(memory_space_id)

    memory_space_name = _read_first_value(
        (resource, memory),
        "memorySpaceName",
        "memory_space_name",
    )
    folder_path = _read_value(memory, "folderPath", "folder_path")
    if not memory_space_name:
        feature = _get_agent_memory_space_feature(agent)
        memory_space_id = _read_value(feature, "memorySpaceId", "memory_space_id")
        if memory_space_id:
            return str(memory_space_id)
        memory_space_name = _read_value(
            feature,
            "memorySpaceName",
            "memory_space_name",
        )
        folder_path = _read_value(feature, "folderPath", "folder_path") or folder_path

    if not memory_space_name:
        return None

    return _resolve_memory_space_id_by_name(str(memory_space_name), folder_path)


def _get_escalation_memory_folder_path(
    resource: AgentEscalationResourceConfig,
    agent: Any | None = None,
) -> str | None:
    """Resolve folder path to use for escalation memory API calls."""
    if not _is_escalation_memory_enabled(resource):
        return None

    memory = _get_escalation_memory_properties(resource)
    memory_space_name = _read_first_value(
        (resource, memory),
        "memorySpaceName",
        "memory_space_name",
    )
    folder_path = _read_value(memory, "folderPath", "folder_path")
    if not memory_space_name and not folder_path:
        feature = _get_agent_memory_space_feature(agent)
        memory_space_name = _read_value(
            feature,
            "memorySpaceName",
            "memory_space_name",
        )
        folder_path = _read_value(feature, "folderPath", "folder_path") or folder_path

    return _resolve_memory_folder_path(
        folder_path, str(memory_space_name) if memory_space_name else None
    )


def _get_escalation_memory_space_name(
    resource: AgentEscalationResourceConfig,
    agent: Any | None = None,
) -> str | None:
    """Resolve memory space name from escalation resource or agent memory feature."""
    if not _is_escalation_memory_enabled(resource):
        return None

    memory = _get_escalation_memory_properties(resource)
    memory_space_name = _read_first_value(
        (resource, memory),
        "memorySpaceName",
        "memory_space_name",
    )
    if memory_space_name:
        return str(memory_space_name)

    feature = _get_agent_memory_space_feature(agent)
    memory_space_name = _read_value(
        feature,
        "memorySpaceName",
        "memory_space_name",
    )
    return str(memory_space_name) if memory_space_name else None


def _get_escalation_memory_settings(
    resource: AgentEscalationResourceConfig,
) -> EscalationMemorySettings | None:
    """Extract memory settings from escalation resource properties."""
    if not _is_escalation_memory_enabled(resource):
        return None

    memory = _get_escalation_memory_properties(resource)
    if memory is None:
        return None
    return _coerce_memory_settings(memory)


def _is_escalation_memory_enabled(resource: AgentEscalationResourceConfig) -> bool:
    memory = _get_escalation_memory_properties(resource)
    memory_enabled = _read_value(memory, "isEnabled", "is_enabled")
    if memory_enabled is not None:
        return bool(memory_enabled)
    return bool(
        _read_value(resource, "isAgentMemoryEnabled", "is_agent_memory_enabled")
    )


def _get_escalation_memory_properties(resource: AgentEscalationResourceConfig) -> Any:
    properties = _read_value(resource, "properties")
    return _read_value(properties, "memory") if properties is not None else None


def _get_agent_memory_space_feature(agent: Any | None) -> Any:
    features = _read_value(agent, "features") or []
    for feature in features:
        feature_type = _read_value(
            feature, "$featureType", "featureType", "feature_type"
        )
        if feature_type != "memorySpace":
            continue
        is_enabled = _read_value(feature, "isEnabled", "is_enabled")
        if is_enabled is False:
            continue
        if _read_value(feature, "memorySpaceId", "memory_space_id") or _read_value(
            feature,
            "memorySpaceName",
            "memory_space_name",
        ):
            return feature
    return None


def _resolve_memory_space_id_by_name(
    memory_space_name: str,
    folder_path: Any,
) -> str | None:
    resolved_folder_path = _resolve_memory_folder_path(folder_path, memory_space_name)
    try:
        escaped_name = memory_space_name.replace("'", "''")
        spaces = UiPath().memory.list(
            filter=f"name eq '{escaped_name}'",
            folder_path=resolved_folder_path,
        )
    except Exception:
        logger.warning(
            "Failed to resolve escalation memory space '%s'",
            memory_space_name,
            exc_info=True,
        )
        return None

    if not spaces.value:
        logger.warning(
            "Escalation memory space '%s' was not found",
            memory_space_name,
        )
        return None
    return str(spaces.value[0].id)


def _resolve_memory_folder_path(
    folder_path: Any,
    memory_space_name: str | None = None,
) -> str | None:
    if memory_space_name:
        folder_path = (
            _get_memory_space_folder_override(memory_space_name) or folder_path
        )
    if folder_path in (None, "", ".", "solution_folder"):
        return get_execution_folder_path()
    return str(folder_path)


def _get_memory_space_folder_override(memory_space_name: str) -> str | None:
    overwrites = _resource_overwrites.get()
    if not overwrites:
        return None

    overwrite = overwrites.get(f"memorySpace.{memory_space_name}")
    if not overwrite:
        return None

    folder_identifier = getattr(overwrite, "folder_identifier", None)
    if not folder_identifier:
        return None

    logger.info(
        "Memory space '%s' folder_path overwritten: '%s'",
        memory_space_name,
        folder_identifier,
    )
    return str(folder_identifier)


def _get_user_email(user: Any) -> str | None:
    """Extract an email address from an Action Center user payload."""
    if user is None:
        return None

    for key in ("emailAddress", "email", "Email", "userName"):
        value = _read_value(user, key)
        if value:
            return str(value)

    return None


def _get_user_id(user: Any) -> str | None:
    """Extract a LLMOps-compatible reviewer ID from an Action Center user payload."""
    if user is None:
        return None

    for key in ("identifier", "userId", "userGlobalId", "id"):
        user_id = _normalize_user_id(_read_value(user, key))
        if user_id is not None:
            return user_id

    return None


async def _resolve_user_id(user: Any) -> str | None:
    """Resolve the Action Center reviewer to the directory ID expected by LLMOps."""
    user_id = _get_user_id(user)
    if user_id:
        return user_id

    email = _get_user_email(user)
    if not email:
        return None

    org_id = UiPathConfig.organization_id
    if not org_id:
        return None

    try:
        response = await UiPath().api_client.request_async(
            "GET",
            f"/identity_/api/Directory/Search/{org_id}",
            scoped="org",
            params={
                "startsWith": email,
                "sourceFilter": ["directoryUsers", "localUsers"],
            },
        )
    except Exception:
        logger.warning("Failed to resolve reviewer '%s'", email, exc_info=True)
        return None

    for entry in response.json() or []:
        if _get_user_email(entry) != email:
            continue
        user_id = _get_user_id(entry)
        if user_id is not None:
            return user_id

    return None


async def _check_escalation_memory_cache(
    memory_space_id: str,
    serialized_input: dict[str, Any],
    folder_path: str | None = None,
    memory_settings: EscalationMemorySettings | None = None,
    memory_space_name: str | None = None,
) -> EscalationMemoryCachedResult | None:
    """Check escalation memory for a cached answer."""
    retriever = EscalationMemoryRetriever(
        memory_space_id,
        memory_space_name=memory_space_name,
        folder_path=folder_path,
        memory_settings=memory_settings,
    )

    try:
        cached_result = await retriever.aretrieve(serialized_input)
    except ValueError as error:
        logger.warning(
            "Skipping escalation memory search for space '%s': %s",
            memory_space_id,
            error,
        )
        _record_custom_metric(MEMORY_CACHE_MISS_METRIC, memory_space_id)
        return None
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
    user_id: str | None = None,
    folder_path: str | None = None,
) -> None:
    """Persist a resolved escalation outcome into memory."""
    normalized_user_id = _normalize_user_id(user_id)
    if user_id is not None and normalized_user_id is None:
        logger.info(
            "Skipping escalation memory reviewer user ID because it is not a GUID: %s",
            user_id,
        )

    try:
        # Keep the OTel import local to match the lookup path and keep this
        # module importable in runtimes where tracing is not installed.
        from opentelemetry import trace as otel_trace

        tracer = otel_trace.get_tracer("uipath_langchain.memory")
    except ImportError:
        tracer = None
        otel_trace = None  # type: ignore[assignment]

    # Span attribute keys match what the LlmOpsHttpExporter and Studio UI expect;
    # "openinference.span.kind" sets the SpanType.
    ingest_span_ctx = (
        tracer.start_as_current_span(
            "Save escalation memory",
            attributes={
                "openinference.span.kind": "agentMemoryWrite",
                "type": "agentMemoryWrite",
                "span_type": "agentMemoryWrite",
                "uipath.custom_instrumentation": True,
                "memorySpaceId": memory_space_id,
                "strategy": ESCALATION_MEMORY_STRATEGY,
            },
        )
        if tracer
        else _noop_context()
    )

    with ingest_span_ctx as ingest_span:
        if ingest_span and hasattr(ingest_span, "set_attribute"):
            ingest_span.set_attribute("fromMemory", False)
        try:
            request = EscalationMemoryIngestRequest(
                span_id=parent_span_id,
                trace_id=trace_id,
                answer=answer,
                attributes=attributes,
                user_id=normalized_user_id,
            )
            sdk = UiPath()
            await sdk.memory.escalation_ingest_async(
                memory_space_id=memory_space_id,
                request=request,
                folder_path=folder_path,
            )
            if ingest_span and hasattr(ingest_span, "set_attribute"):
                ingest_span.set_attribute("savedToMemory", True)
            logger.info(
                "Ingested escalation outcome into memory space '%s'", memory_space_id
            )
        except Exception as error:
            if ingest_span and hasattr(ingest_span, "set_attribute"):
                ingest_span.set_attribute("savedToMemory", False)
            if otel_trace and ingest_span and hasattr(ingest_span, "set_status"):
                ingest_span.set_status(otel_trace.StatusCode.ERROR, repr(error))
            if ingest_span and hasattr(ingest_span, "record_exception"):
                ingest_span.record_exception(error)
            logger.warning(
                "Failed to ingest escalation outcome into memory space '%s': %s",
                memory_space_id,
                error,
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


def _serialize_search_response_for_trace(response: Any) -> str:
    if isinstance(response, BaseModel):
        response = response.model_dump(by_alias=True, exclude_none=True)
    return _json_dumps(response)


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 0


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=str)


def _cached_result_from_search_response(
    response: Any,
) -> EscalationMemoryCachedResult | None:
    results = _read_value(response, "results") or []
    if not results:
        return None

    answer = _read_value(results[0], "answer")
    if not answer:
        return None

    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            logger.warning("Escalation memory cache entry answer is not valid JSON")
            return None

    output = _read_value(answer, "output", "Output")
    if output is None:
        logger.warning(
            "Escalation memory cache entry has no 'output' property; treating as cache miss."
        )
        return None

    return EscalationMemoryCachedResult(
        output=output,
        outcome=_read_value(answer, "outcome", "Outcome"),
    )


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
        value = _read_mapping_value(source, keys)
        return None if value is _MISSING_VALUE else value
    if isinstance(source, BaseModel):
        value = _read_mapping_value(source.model_extra or {}, keys)
        if value is not _MISSING_VALUE:
            return value
    return _read_attribute_value(source, keys)


def _read_first_value(sources: tuple[Any, ...], *keys: str) -> Any:
    for source in sources:
        value = _read_value(source, *keys)
        if value is not None:
            return value
    return None


def _read_mapping_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return _MISSING_VALUE


def _read_attribute_value(source: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = getattr(source, key, _MISSING_VALUE)
        if value is not _MISSING_VALUE:
            return value
    return None


def _normalize_user_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError):
        return None


def _stringify_search_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float | list | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)
