"""State initialization node for the ReAct Agent graph."""

import logging
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite
from pydantic import BaseModel

from .job_attachments import (
    get_job_attachments,
    parse_attachments_from_conversation_messages,
)
from .types import AgentResources

logger = logging.getLogger(__name__)


async def _build_schema_context(entities: list) -> str:
    """Build schema context, using ECP enrichment if feature flag is enabled."""
    from uipath_langchain.agent.tools.datafabric_tool import (
        format_schemas_for_context,
    )

    ecp_enabled = False
    try:
        from uipath.core.feature_flags import FeatureFlags

        ecp_enabled = FeatureFlags.is_flag_enabled(
            "EnableEntityContextPackEnrichment"
        )
    except Exception:
        logger.info("Feature flags unavailable, using basic schema context")

    if ecp_enabled:
        try:
            from uipath_langchain.agent.tools.datafabric_tool import (
                build_entity_context_packs,
                format_ecp_for_context,
            )

            logger.info("Building enriched Entity Context Packs")
            context_packs = await build_entity_context_packs(entities)
            return format_ecp_for_context(context_packs)
        except Exception:
            logger.warning(
                "ECP enrichment failed, falling back to basic schema",
                exc_info=True,
            )

    return format_schemas_for_context(entities)


def create_init_node(
    messages: Sequence[SystemMessage | HumanMessage]
    | Callable[..., Sequence[SystemMessage | HumanMessage]],
    input_schema: type[BaseModel] | None,
    is_conversational: bool = False,
    resources_for_init: AgentResources | None = None,
):
    async def graph_state_init(state: Any) -> Any:
        # --- Data Fabric schema fetch (INIT-time) ---
        schema_context: str | None = None
        if resources_for_init:
            from uipath_langchain.agent.tools.datafabric_tool import (
                fetch_entity_schemas,
                get_datafabric_entity_identifiers_from_resources,
            )

            entity_identifiers = get_datafabric_entity_identifiers_from_resources(
                resources_for_init
            )
            if entity_identifiers:
                logger.info(
                    "Fetching Data Fabric schemas for %d identifier(s)",
                    len(entity_identifiers),
                )
                entities = await fetch_entity_schemas(entity_identifiers)
                schema_context = await _build_schema_context(entities)

        # --- Resolve messages ---
        resolved_messages: Sequence[SystemMessage | HumanMessage] | Overwrite
        if callable(messages):
            if schema_context:
                resolved_messages = list(
                    messages(state, additional_context=schema_context)
                )
            else:
                resolved_messages = list(messages(state))
        else:
            resolved_messages = list(messages)

        # Log the full system prompt for debugging
        _msgs = (
            resolved_messages.value
            if isinstance(resolved_messages, Overwrite)
            else resolved_messages
        )
        for _m in _msgs:
            if isinstance(_m, SystemMessage):
                logger.debug("Full system prompt:\n%s", _m.content)
                with open("/tmp/init_node_full_system_prompt.txt", "w") as _fp:
                    _fp.write(str(_m.content))
                break

        if is_conversational:
            # For conversational agents we need to reorder the messages so that the system message is first, followed by
            # the initial user message. When resuming the conversation, the state will have the entire message history,
            # including the system message. In this case, we need to replace the system message from the state with the
            # newly generated one. It will have the current date/time and reflect any changes to user settings. The add
            # reducer is used for the messages property in the state, so by default new messages are appended to the end
            # and using Overwrite will cause LangGraph to replace the entire array instead.
            if len(state.messages) > 0 and isinstance(state.messages[0], SystemMessage):
                preserved_messages = state.messages[1:]
            else:
                preserved_messages = state.messages
            resolved_messages = Overwrite([*resolved_messages, *preserved_messages])

        schema = input_schema if input_schema is not None else BaseModel
        job_attachments = get_job_attachments(schema, state)
        job_attachments_dict = {
            str(att.id): att for att in job_attachments if att.id is not None
        }
        # Merge attachments from preserved messages for conversational agents
        if is_conversational:
            message_attachments = parse_attachments_from_conversation_messages(
                preserved_messages
            )
            job_attachments_dict.update(message_attachments)

        # Calculate initial message count for tracking new messages
        initial_message_count = (
            len(resolved_messages.value)
            if isinstance(resolved_messages, Overwrite)
            else len(resolved_messages)
        )

        return {
            "messages": resolved_messages,
            "inner_state": {
                "job_attachments": job_attachments_dict,
                "initial_message_count": initial_message_count,
            },
        }

    return graph_state_init
