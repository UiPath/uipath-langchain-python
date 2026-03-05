"""State initialization node for the ReAct Agent graph."""

import logging
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite
from pydantic import BaseModel

from .job_attachments import (
    get_job_attachments,
)
from .types import AgentResources, AgentSettings

logger = logging.getLogger(__name__)


def create_init_node(
    messages: Sequence[SystemMessage | HumanMessage]
    | Callable[..., Sequence[SystemMessage | HumanMessage]],
    input_schema: type[BaseModel] | None,
    is_conversational: bool = False,
    agent_settings: AgentSettings | None = None,
    resources_for_init: AgentResources | None = None,
):
    async def graph_state_init(state: Any) -> Any:
        # --- Data Fabric schema fetch (INIT-time) ---
        schema_context: str | None = None
        if resources_for_init:
            from uipath_langchain.agent.tools.datafabric_tool import (
                fetch_entity_schemas,
                format_schemas_for_context,
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
                schema_context = format_schemas_for_context(entities)

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

        return {
            "messages": resolved_messages,
            "inner_state": {
                "job_attachments": job_attachments_dict,
                "agent_settings": agent_settings,
            },
        }

    return graph_state_init
