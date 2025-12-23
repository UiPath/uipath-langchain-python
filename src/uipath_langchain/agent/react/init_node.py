"""State initialization node for the ReAct Agent graph."""

from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .job_attachments import (
    get_job_attachments,
)


def create_init_node(
    messages: Sequence[SystemMessage | HumanMessage]
    | Callable[[Any], Sequence[SystemMessage | HumanMessage]],
    input_schema: type[BaseModel] | None,
):
    def graph_state_init(state: Any):
        if callable(messages):
            resolved_messages = messages(state)
        else:
            resolved_messages = messages

        schema = input_schema if input_schema is not None else BaseModel
        job_attachments = get_job_attachments(schema, state)
        job_attachments_dict = {
            str(att.id): att for att in job_attachments if att.id is not None
        }

        return {
            "messages": list(resolved_messages),
            "job_attachments": job_attachments_dict,
        }

    return graph_state_init
