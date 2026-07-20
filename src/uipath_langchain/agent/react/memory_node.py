"""Memory recall node for Agent Episodic Memory.

Queries the UiPath Memory service (via LLMOps) for similar past episodes
and stores the server-generated systemPromptInjection in graph state so
the INIT node can append it to the system prompt.

When a memory space enables file analysis (space-owned settings), the node
also derives an ephemeral ``fileText_query`` from each key-included attachment
via one multimodal LLM call per field and searches with that text instead of
the raw attachment reference. The derived text is query-only — it is never
persisted and never written back into the agent's input/state, so the ReAct
loop and the analyze-files tool still receive the full attachment.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from uipath.platform import UiPath
from uipath.platform.errors import EnrichedException
from uipath.platform.memory import (
    FieldSettings,
    MemorySearchRequest,
    SearchField,
    SearchMode,
    SearchSettings,
)

from ...chat import get_chat_model
from ...chat.helpers import extract_text_content
from ..tools.internal_tools.analyze_files_tool import (
    add_files_to_message,
    resolve_attachments_to_file_infos,
)
from .job_attachments import get_job_attachment_paths
from .types import MemoryConfig
from .utils import extract_input_data_from_state

logger = logging.getLogger(__name__)

# Field type prefix used for episodic-memory keyPaths (["agent-input", <field>]).
_ANALYSIS_FIELD_TYPE = "agent-input"
# Cap the derived query text so a large document can't blow up the search request.
_FILE_TEXT_MAX_CHARS = 8000

_UNSET = object()


@contextmanager
def _noop_context():
    """No-op context manager when OTel is unavailable."""
    yield None


def _get_tracer() -> tuple[Any, Any]:
    """Return ``(tracer, otel_trace)`` or ``(None, None)`` when OTel is unavailable."""
    try:
        from opentelemetry import trace as otel_trace

        return otel_trace.get_tracer("uipath_langchain.memory"), otel_trace
    except ImportError:
        return None, None


def create_memory_recall_node(
    memory_config: MemoryConfig,
    input_schema: type[BaseModel] | None = None,
    model: BaseChatModel | None = None,
):
    """Create an async graph node that retrieves memory injection.

    The node queries ``sdk.memory.search_async()`` and writes the
    ``systemPromptInjection`` string into ``inner_state.memory_injection``.
    On failure it logs a warning and continues with an empty injection.

    Args:
        memory_config: Memory configuration with space ID and search settings.
        input_schema: The agent's input schema, used to locate attachment fields.
        model: The agent's chat model. Required for file-key analysis; when
            ``None`` the file-analysis pre-step is skipped entirely (today's
            text-only recall behavior).

    Returns:
        An async callable suitable for ``builder.add_node()``.
    """

    # Per-run caches on the node closure (fetched at most once).
    _settings_cache: list[Any] = [_UNSET]
    _analysis_model_cache: list[Any] = [_UNSET]

    async def _get_settings() -> dict[str, Any] | None:
        if _settings_cache[0] is _UNSET:
            _settings_cache[0] = await _fetch_space_settings(
                memory_config.memory_space_id
            )
        return _settings_cache[0]

    def _get_analysis_model(settings: dict[str, Any]) -> BaseChatModel:
        if _analysis_model_cache[0] is _UNSET:
            _analysis_model_cache[0] = _build_analysis_model(
                model, settings.get("analysisModel")
            )
        return _analysis_model_cache[0]

    async def memory_recall_node(state: Any) -> dict[str, Any]:
        input_model = input_schema if input_schema is not None else BaseModel
        input_arguments = extract_input_data_from_state(state, input_model)
        if not input_arguments:
            logger.debug("Memory recall: no user inputs found in state")
            return {}

        tracer, otel_trace = _get_tracer()

        # File-key analysis pre-step (query-side, ephemeral). Soft-fails to the
        # text-only behavior on any error. Never mutates state/input — only the
        # local search arguments are substituted.
        search_arguments = input_arguments
        if model is not None:
            settings = await _get_settings()
            if settings:
                analyzed, skipped = await _analyze_file_fields_for_search(
                    input_arguments=input_arguments,
                    input_model=input_model,
                    settings=settings,
                    field_weights=memory_config.field_weights or None,
                    build_analysis_model=lambda: _get_analysis_model(settings),
                    tracer=tracer,
                )
                if analyzed or skipped:
                    search_arguments = dict(input_arguments)
                    search_arguments.update(analyzed)
                    for field in skipped:
                        # Enabled file field with no usable analysis text: drop it
                        # entirely rather than searching on the raw attachment ref.
                        search_arguments.pop(field, None)

        fields = _build_search_fields(
            search_arguments, field_weights=memory_config.field_weights or None
        )
        if not fields:
            logger.debug(
                "Memory recall: no search fields after filtering (inputs=%s, weights=%s)",
                list(input_arguments.keys()),
                memory_config.field_weights,
            )
            return {}

        request = MemorySearchRequest(
            fields=fields,
            settings=SearchSettings(
                threshold=memory_config.threshold,
                result_count=memory_config.result_count,
                search_mode=SearchMode.Hybrid,
            ),
            definition_system_prompt="",
        )

        results_count = 0
        # Wrap the search in OTel spans so "Find previous memories" and
        # "Apply dynamic few shot" appear in the Execution Trace with
        # correct timing. The LlmOpsHttpExporter picks these up.
        injection = ""

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
                    "memorySpaceName": memory_config.memory_space_name or "",
                    "memorySpaceId": memory_config.memory_space_id,
                    "strategy": "DynamicFewShotPrompt",
                },
            )
            if tracer
            else _noop_context()
        )

        with lookup_span_ctx as lookup_span:
            fewshot_span_ctx = (
                tracer.start_as_current_span(
                    "Apply dynamic few shot",
                    attributes={
                        "openinference.span.kind": "applyDynamicFewShot",
                        "type": "applyDynamicFewShot",
                        "span_type": "applyDynamicFewShot",
                        "uipath.custom_instrumentation": True,
                        "memorySpaceName": memory_config.memory_space_name or "",
                        "memorySpaceId": memory_config.memory_space_id,
                    },
                )
                if tracer
                else _noop_context()
            )

            with fewshot_span_ctx as fewshot_span:
                try:
                    sdk = UiPath()
                    folder_key = memory_config.folder_key
                    if not folder_key and memory_config.folder_path:
                        folder_key = await sdk.folders.retrieve_folder_key_async(
                            memory_config.folder_path
                        )
                    response = await sdk.memory.search_async(
                        memory_space_id=memory_config.memory_space_id,
                        request=request,
                        folder_key=folder_key,
                    )
                    injection = response.system_prompt_injection
                    results_count = len(response.results)
                    logger.info(
                        "Memory recall returned %d results for space '%s'",
                        results_count,
                        memory_config.memory_space_id,
                    )
                    # Set request/response on fewshot span as JSON strings.
                    # The exporter parses JSON strings back to objects.
                    # The UI reads "response" to display matched memory items.
                    if fewshot_span and hasattr(fewshot_span, "set_attribute"):
                        import json

                        fewshot_span.set_attribute(
                            "request",
                            json.dumps(
                                request.model_dump(by_alias=True, exclude_none=True)
                            ),
                        )
                        fewshot_span.set_attribute(
                            "response",
                            json.dumps(
                                response.model_dump(by_alias=True, exclude_none=True)
                            ),
                        )
                except Exception as e:
                    if isinstance(e, EnrichedException):
                        error_detail = (
                            f"{e} | status={e.status_code} body={e.response_content}"
                        )
                    else:
                        error_detail = repr(e)
                    logger.warning(
                        "Memory recall failed for space '%s': %s",
                        memory_config.memory_space_id,
                        error_detail,
                    )
                    if otel_trace:
                        if fewshot_span and hasattr(fewshot_span, "set_status"):
                            fewshot_span.set_status(
                                otel_trace.StatusCode.ERROR, error_detail
                            )
                        if lookup_span and hasattr(lookup_span, "set_status"):
                            lookup_span.set_status(
                                otel_trace.StatusCode.ERROR, error_detail
                            )

            # Set result attributes after search completes
            if lookup_span and hasattr(lookup_span, "set_attribute"):
                lookup_span.set_attribute("memoryItemsMatched", results_count)
                if injection:
                    lookup_span.set_attribute("result", injection)

        if not injection:
            return {}

        return {"inner_state": {"memory_injection": injection}}

    return memory_recall_node


async def _fetch_space_settings(space_id: str) -> dict[str, Any] | None:
    """Fetch a memory space's settings from LLMOps.

    Uses the SDK's low-level API client so we get bearer auth, org/tenant
    scoping and retries for free (no SDK release needed). Soft-fails to ``None``
    on any error so recall proceeds with no file analysis (today's behavior).
    """
    try:
        sdk = UiPath()
        response = await sdk.api_client.request_async(
            "GET",
            f"/llmopstenant_/api/Memory/{space_id}/settings",
            include_folder_headers=True,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(
            "Memory file-analysis: failed to fetch space settings for '%s': %s",
            space_id,
            repr(e),
        )
        return None


def _build_analysis_model(
    agent_model: BaseChatModel, analysis_model_name: str | None
) -> BaseChatModel:
    """Build the (non-streaming) model used for file-key analysis.

    Prefers the space-pinned ``analysisModel`` so the query side matches the
    ingest side (symmetry: same prompt + same model in, same key out). Falls
    back to a non-streaming copy of the agent's own model if the pinned model
    cannot be constructed.
    """
    if analysis_model_name:
        try:
            fresh = get_chat_model(analysis_model_name)
            return fresh.model_copy(update={"disable_streaming": True})
        except Exception as e:
            logger.warning(
                "Memory file-analysis: could not build analysis model '%s' (%s); "
                "falling back to the agent model. "
                "TODO(POC): honor settings.analysisModel for symmetry.",
                analysis_model_name,
                repr(e),
            )
    return agent_model.model_copy(update={"disable_streaming": True})


def _top_level_attachment_fields(model: type[BaseModel]) -> set[str]:
    """Return the names of top-level attachment fields (single or list) in ``model``.

    Scope is intentionally limited to flat memory keyPaths: only ``$.field`` and
    ``$.field[*]`` qualify; nested/array-of-object paths are out of scope.
    """
    fields: set[str] = set()
    for path in get_job_attachment_paths(model):
        name = _top_level_attachment_field(path)
        if name:
            fields.add(name)
    return fields


def _top_level_attachment_field(json_path: str) -> str | None:
    """Extract the field name from a top-level attachment JSONPath, else ``None``.

    ``$.doc`` -> ``"doc"``; ``$.docs[*]`` -> ``"docs"``; nested paths such as
    ``$.a.b`` or ``$.a[*][*]`` return ``None``.
    """
    if not json_path.startswith("$."):
        return None
    rest = json_path[2:]
    if rest.endswith("[*]"):
        rest = rest[:-3]
    if not rest or "." in rest or "[" in rest:
        return None
    return rest


def _enabled_file_prompts(
    settings: dict[str, Any], field_weights: dict[str, float] | None
) -> dict[str, str]:
    """Map field name -> analysis prompt for enabled, in-scope fileAnalysis entries.

    Only entries that are enabled, carry a flat ``["agent-input", field]`` keyPath
    with a non-empty prompt, and target a field the recall search actually uses
    (present in ``field_weights``) are returned.
    """
    result: dict[str, str] = {}
    for entry in settings.get("fileAnalysis") or []:
        if not entry.get("enabled"):
            continue
        key_path = entry.get("keyPath") or []
        if len(key_path) != 2 or key_path[0] != _ANALYSIS_FIELD_TYPE:
            continue
        field = key_path[1]
        prompt = entry.get("analysisPrompt") or ""
        if not prompt:
            continue
        if field_weights and field not in field_weights:
            continue
        result[field] = prompt
    return result


def _normalize_attachment_items(value: Any) -> list[Any]:
    """Normalize a single attachment or a list of attachments to a list of truthy items."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [item for item in items if item]


def _render_analysis_prompt(prompt: str, agent_prompt_reference: str | None) -> str:
    """Fill the ``{FILE}`` and ``{AGENT_PROMPT}`` placeholders in an analysis prompt.

    ``{FILE}`` always becomes "the attached file". ``{AGENT_PROMPT}`` becomes the
    space's agent-prompt reference when present, otherwise it is removed.
    """
    rendered = prompt.replace("{FILE}", "the attached file")
    rendered = rendered.replace("{AGENT_PROMPT}", agent_prompt_reference or "")
    return rendered


async def _analyze_file_fields_for_search(
    *,
    input_arguments: dict[str, Any],
    input_model: type[BaseModel],
    settings: dict[str, Any],
    field_weights: dict[str, float] | None,
    build_analysis_model: Callable[[], BaseChatModel],
    tracer: Any,
) -> tuple[dict[str, str], set[str]]:
    """Derive ephemeral query text for key-included attachment fields.

    Returns ``(analyzed, skipped)`` where ``analyzed`` maps a field name to its
    derived text and ``skipped`` is the set of enabled file fields that produced
    no usable text (and must be dropped from the search entirely). Each field is
    soft-failed independently — an error on one never aborts the others.
    """
    enabled_prompts = _enabled_file_prompts(settings, field_weights)
    if not enabled_prompts:
        return {}, set()

    attachment_fields = _top_level_attachment_fields(input_model)
    candidates = {
        field: prompt
        for field, prompt in enabled_prompts.items()
        if field in attachment_fields
    }
    if not candidates:
        return {}, set()

    analysis_model = build_analysis_model()
    agent_prompt_reference = settings.get("agentPromptReference")

    analyzed: dict[str, str] = {}
    skipped: set[str] = set()
    for field, prompt in candidates.items():
        items = _normalize_attachment_items(input_arguments.get(field))
        if not items:
            # Field declared but no attachment supplied this run — nothing to do;
            # leave it to the normal (empty) filtering rather than skip-listing it.
            continue
        try:
            text = await _analyze_single_field(
                field=field,
                prompt=prompt,
                items=items,
                agent_prompt_reference=agent_prompt_reference,
                analysis_model=analysis_model,
                memory_space_id=str(settings.get("memorySpaceId") or ""),
                tracer=tracer,
            )
        except Exception as e:
            logger.warning(
                "Memory file-analysis: skipping field '%s': %s", field, repr(e)
            )
            skipped.add(field)
            continue
        if text:
            analyzed[field] = text
        else:
            skipped.add(field)

    return analyzed, skipped


async def _analyze_single_field(
    *,
    field: str,
    prompt: str,
    items: list[Any],
    agent_prompt_reference: str | None,
    analysis_model: BaseChatModel,
    memory_space_id: str,
    tracer: Any,
) -> str:
    """Run one multimodal analysis call for a single attachment field.

    Returns the derived text (stripped and truncated), or ``""`` when there is
    nothing usable to search on.
    """
    file_infos = await resolve_attachments_to_file_infos(items)
    if not file_infos:
        return ""

    # TODO(POC): PII-mask via PiiMasker before the LLM call — files are sent to
    # the gateway unmasked in the POC (same posture as the analyze-files tool
    # without a policy). Must mask identically to the ingest side before GA.
    system_prompt = _render_analysis_prompt(prompt, agent_prompt_reference)
    human_message = await add_files_to_message(HumanMessage(content=""), file_infos)
    messages = [SystemMessage(content=system_prompt), human_message]

    span_ctx = (
        tracer.start_as_current_span(
            "Analyze file for memory key",
            attributes={
                "uipath.custom_instrumentation": True,
                "memoryKeyField": field,
                "memorySpaceId": memory_space_id,
            },
        )
        if tracer
        else _noop_context()
    )

    start = time.perf_counter()
    with span_ctx as span:
        try:
            result = await analysis_model.ainvoke(messages)
        finally:
            if span is not None and hasattr(span, "set_attribute"):
                span.set_attribute("durationMs", (time.perf_counter() - start) * 1000.0)

    # The trace carries only the raw attachment ref (via the agent input) — never
    # the derived text — so we deliberately do NOT stamp fileText on the span.
    return extract_text_content(result).strip()[:_FILE_TEXT_MAX_CHARS]


def _build_search_fields(
    input_arguments: dict[str, Any],
    field_weights: dict[str, float] | None = None,
    field_type: str = "agent-input",
) -> list[SearchField]:
    """Convert agent input arguments to SearchField objects.

    The key_path must be prefixed with the field type:
      keyPath = [fieldType, fieldName]
    e.g. ["agent-input", "a"] for episodic memory.
    """
    fields: list[SearchField] = []
    for name, value in input_arguments.items():
        value_str = str(value) if value is not None else ""
        if not value_str or name.startswith("uipath__"):
            continue
        # When field_weights is specified, only include fields with configured weights
        if field_weights and name not in field_weights:
            continue
        settings = FieldSettings()
        if field_weights and name in field_weights:
            settings = FieldSettings(weight=field_weights[name])
        fields.append(
            SearchField(key_path=[field_type, name], value=value_str, settings=settings)
        )
    return fields
