"""Termination node for the Agent graph."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from pydantic import BaseModel
from uipath._cli._runtime._contracts import UiPathErrorCode
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from .exceptions import (
    AgentNodeRoutingException,
    AgentTerminationException,
)
from .types import AgentGraphState


def create_terminate_node(
    response_schema: type[BaseModel] | None = None,
):
    """Validates and extracts end_execution args to state output field."""

    def terminate_node(state: AgentGraphState):
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise AgentNodeRoutingException(
                f"Expected last message to be AIMessage, got {type(last_message).__name__}"
            )

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]

            if tool_name == END_EXECUTION_TOOL.name:
                args = tool_call["args"]
                output_schema = response_schema or END_EXECUTION_TOOL.args_schema
                validated = output_schema.model_validate(args)
                return validated.model_dump()

            if tool_name == RAISE_ERROR_TOOL.name:
                error_message = tool_call["args"].get(
                    "message", "The LLM did not set the error message"
                )
                detail = tool_call["args"].get("details", "")
                raise AgentTerminationException(
                    code=UiPathErrorCode.EXECUTION_ERROR,
                    title=error_message,
                    detail=detail,
                )

        raise AgentNodeRoutingException(
            "No control flow tool call found in terminate node. Unexpected state."
        )

    return terminate_node
