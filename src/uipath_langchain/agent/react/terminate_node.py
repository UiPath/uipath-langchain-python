"""Termination node for the Agent graph."""

from __future__ import annotations

from typing import Any, NoReturn

from langchain_core.messages import AIMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.runtime.errors import UiPathErrorCode

from ..exceptions import (
    AgentNodeRoutingException,
    AgentTerminationException,
)
from .types import AgentGraphState, AgentTermination


def _handle_end_execution(
    args: dict[str, Any], response_schema: type[BaseModel] | None
) -> dict[str, Any]:
    """Handle LLM-initiated termination via END_EXECUTION_TOOL."""
    output_schema = response_schema or END_EXECUTION_TOOL.args_schema
    validated = output_schema.model_validate(args)
    return validated.model_dump()


def _handle_raise_error(args: dict[str, Any]) -> NoReturn:
    """Handle LLM-initiated error via RAISE_ERROR_TOOL."""
    error_message = args.get("message", "The LLM did not set the error message")
    detail = args.get("details", "")
    raise AgentTerminationException(
        code=UiPathErrorCode.EXECUTION_ERROR,
        title=error_message,
        detail=detail,
    )


def _handle_agent_termination(termination: AgentTermination) -> NoReturn:
    """Handle Command-based termination."""
    raise AgentTerminationException(
        code=UiPathErrorCode.EXECUTION_ERROR,
        title=termination.title,
        detail=termination.detail,
    )


def create_terminate_node(
    response_schema: type[BaseModel] | None = None,
):
    """Handles Agent Graph termination for multiple sources and output or error propagation to Orchestrator.

    Termination scenarios:
    1. Command based termination with information in state (e.g: escalation)
    2. LLM-initiated termination (END_EXECUTION_TOOL)
    3. LLM-initiated error (RAISE_ERROR_TOOL)
    """

    def terminate_node(state: AgentGraphState):
        if state.termination:
            _handle_agent_termination(state.termination)

        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise AgentNodeRoutingException(
                f"Expected last message to be AIMessage, got {type(last_message).__name__}"
            )

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]

            if tool_name == END_EXECUTION_TOOL.name:
                return _handle_end_execution(tool_call["args"], response_schema)

            if tool_name == RAISE_ERROR_TOOL.name:
                _handle_raise_error(tool_call["args"])

        raise AgentNodeRoutingException(
            "No control flow tool call found in terminate node. Unexpected state."
        )

    return terminate_node
