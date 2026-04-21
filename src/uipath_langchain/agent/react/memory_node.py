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

    async def memory_recall_node(state: AgentGraphState) -> dict[str, Any]:
        input_arguments = _extract_user_inputs(state)
        if not input_arguments:
            return {}

        fields = _build_search_fields(input_arguments)
        if not fields:
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
            response = await sdk.memory.search_async(
                memory_space_id=memory_config.memory_space_id,
                request=request,
                folder_key=memory_config.folder_key,
            )
            injection = response.system_prompt_injection
            logger.info(
                "Memory recall returned %d results for space '%s'",
                len(response.results),
                memory_config.memory_space_id,
            )
        except Exception:
            logger.warning(
                "Memory recall failed for space '%s', continuing without injection",
                memory_config.memory_space_id,
                exc_info=True,
            )
            injection = ""

        if not injection:
            return {}

        return {"inner_state": {"memory_injection": injection}}

    return memory_recall_node


def _extract_user_inputs(state: AgentGraphState) -> dict[str, Any]:
    """Extract user-defined input fields from graph state, excluding internal fields."""
    internal_fields = set(AgentGraphState.model_fields.keys())
    if isinstance(state, dict):
        return {k: v for k, v in state.items() if k not in internal_fields}
    return {
        k: v
        for k, v in state.model_dump().items()
        if k not in internal_fields and v is not None
    }


def _build_search_fields(
    input_arguments: dict[str, Any],
    field_weights: dict[str, float] | None = None,
) -> list[SearchField]:
    """Convert agent input arguments to SearchField objects."""
    fields: list[SearchField] = []
    for name, value in input_arguments.items():
        if value is None or name.startswith("uipath__"):
            continue
        settings = FieldSettings()
        if field_weights and name in field_weights:
            settings = FieldSettings(weight=field_weights[name])
        fields.append(SearchField(key_path=[name], value=str(value), settings=settings))
    return fields
