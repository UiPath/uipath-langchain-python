"""State initialization node for the ReAct Agent graph."""

from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite
from pydantic import BaseModel

from .job_attachments import (
    get_job_attachments,
)


def create_init_node(
    messages: Sequence[SystemMessage | HumanMessage]
    | Callable[[Any], Sequence[SystemMessage | HumanMessage]],
    input_schema: type[BaseModel] | None,
    is_conversational: bool = False,
):
    def graph_state_init(state: Any) -> Any:
        resolved_messages: Sequence[SystemMessage | HumanMessage] | Overwrite
        if callable(messages):
            resolved_messages = list(messages(state))
        else:
            resolved_messages = list(messages)
        if is_conversational:
            # For conversational agents we need to reorder the messages so that the system message is first, followed by
            # the initial user message. The initial user message is put in the state by UiPathLangGraphRuntime. The add
            # reducer is used for the messages property in the state, so by default new messages are appended to the end.
            resolved_messages = Overwrite([*resolved_messages, *state.messages])

        schema = input_schema if input_schema is not None else BaseModel
        job_attachments = get_job_attachments(schema, state)
        job_attachments_dict = {
            str(att.id): att for att in job_attachments if att.id is not None
        }

        return {
            "messages": resolved_messages,
            "inner_state": {
                "job_attachments": job_attachments_dict,
            },
        }

    return graph_state_init
