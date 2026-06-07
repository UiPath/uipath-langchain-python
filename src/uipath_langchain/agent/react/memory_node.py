"""Memory recall node for Agent Episodic Memory.

Queries the UiPath Memory service (via LLMOps) for similar past episodes
and stores the server-generated systemPromptInjection in graph state so
the INIT node can append it to the system prompt.
"""

import logging
from contextlib import contextmanager
from typing import Any

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

from .types import MemoryConfig
from .utils import extract_input_data_from_state

logger = logging.getLogger(__name__)


@contextmanager
def _noop_context():
    """No-op context manager when OTel is unavailable."""
    yield None


def create_memory_recall_node(
    memory_config: MemoryConfig,
    input_schema: type[BaseModel] | None = None,
):
    """Create an async graph node that retrieves memory injection.

    The node queries ``sdk.memory.search_async()`` and writes the
    ``systemPromptInjection`` string into ``inner_state.memory_injection``.
    On failure it logs a warning and continues with an empty injection.

    Args:
        memory_config: Memory configuration with space ID and search settings.

    Returns:
        An async callable suitable for ``builder.add_node()``.
    """

    async def memory_recall_node(state: Any) -> dict[str, Any]:
        input_model = input_schema if input_schema is not None else BaseModel
        input_arguments = extract_input_data_from_state(state, input_model)
        if not input_arguments:
            logger.debug("Memory recall: no user inputs found in state")
            return {}

        fields = _build_search_fields(
            input_arguments, field_weights=memory_config.field_weights or None
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
        try:
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
