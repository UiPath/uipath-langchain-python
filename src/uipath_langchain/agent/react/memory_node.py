"""Memory recall node for Agent Episodic Memory.

Queries the UiPath Memory service (via LLMOps) for similar past episodes
and stores the server-generated systemPromptInjection in graph state so
the INIT node can append it to the system prompt.
"""

import logging
from typing import Any

from uipath.platform import UiPath
from uipath.platform.memory import (
    FieldSettings,
    MemorySearchRequest,
    SearchField,
    SearchMode,
    SearchSettings,
)

from .types import AgentGraphState, MemoryConfig

logger = logging.getLogger(__name__)


def create_memory_recall_node(
    memory_config: MemoryConfig,
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
        input_arguments = _extract_user_inputs(state)
        if not input_arguments:
            logger.warning("Memory recall: no user inputs found in state")
            return {}

        fields = _build_search_fields(
            input_arguments, field_weights=memory_config.field_weights or None
        )
        if not fields:
            logger.warning(
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
        )

        try:
            sdk = UiPath()
            # Resolve folder_key: explicit > resolve from folder_path > SDK default
            folder_key = memory_config.folder_key
            if not folder_key and memory_config.folder_path:
                folder_key = sdk.folders.retrieve_folder_key(
                    memory_config.folder_path
                )
            logger.warning(
                "Memory recall: searching space='%s', folder_key='%s', "
                "fields=%s, threshold=%s, result_count=%s",
                memory_config.memory_space_id,
                folder_key,
                [(f.key_path, f.value) for f in fields],
                memory_config.threshold,
                memory_config.result_count,
            )
            response = await sdk.memory.search_async(
                memory_space_id=memory_config.memory_space_id,
                request=request,
                folder_key=folder_key,
            )
            injection = response.system_prompt_injection
            logger.warning(
                "Memory recall returned %d results for space '%s'",
                len(response.results),
                memory_config.memory_space_id,
            )
        except Exception as e:
            # Try to extract HTTP response body from the exception chain
            error_detail = repr(e)
            for exc in [e, getattr(e, "__cause__", None), getattr(e, "__context__", None)]:
                if exc and hasattr(exc, "response"):
                    try:
                        resp = exc.response  # type: ignore[union-attr]
                        error_detail = f"{exc} | status={resp.status_code} body={resp.text}"
                    except Exception:
                        pass
                    break
            logger.warning(
                "Memory recall failed for space '%s': %s",
                memory_config.memory_space_id,
                error_detail,
            )
            injection = ""

        if not injection:
            return {}

        return {"inner_state": {"memory_injection": injection}}

    return memory_recall_node


def _extract_user_inputs(state: Any) -> dict[str, Any]:
    """Extract user-defined input fields from graph state, excluding internal fields.

    Handles both dict states (LangGraph internal) and Pydantic model states.
    For Pydantic models, uses the runtime type's model_dump() and also
    checks __dict__ for fields that model_dump() may miss when the state
    is deserialized as the base AgentGraphState class.
    """
    internal_fields = set(AgentGraphState.model_fields.keys())
    if isinstance(state, dict):
        state_data = state
    elif hasattr(state, "model_dump"):
        # Start with model_dump(), then merge any extra fields from __dict__
        # that model_dump() missed (happens when LangGraph deserializes
        # CompleteAgentGraphState as base AgentGraphState)
        state_data = state.model_dump()
        for k, v in state.__dict__.items():
            if k not in state_data and not k.startswith("_"):
                state_data[k] = v
    else:
        state_data = {}
    return {
        k: v
        for k, v in state_data.items()
        if k not in internal_fields and v is not None
    }


def _build_search_fields(
    input_arguments: dict[str, Any],
    field_weights: dict[str, float] | None = None,
    field_type: str = "agent-input",
) -> list[SearchField]:
    """Convert agent input arguments to SearchField objects.

    The key_path must be prefixed with the field type, matching the
    Temporal backend's FieldBuilder (FieldBuilder.cs:15):
      keyPath = [fieldType.StringValue(), fieldName]
    e.g. ["agent-input", "a"] for episodic memory.
    """
    fields: list[SearchField] = []
    for name, value in input_arguments.items():
        if value is None or name.startswith("uipath__"):
            continue
        # When field_weights is specified, only include fields with configured weights
        if field_weights and name not in field_weights:
            continue
        settings = FieldSettings()
        if field_weights and name in field_weights:
            settings = FieldSettings(weight=field_weights[name])
        fields.append(
            SearchField(
                key_path=[field_type, name], value=str(value), settings=settings
            )
        )
    return fields
