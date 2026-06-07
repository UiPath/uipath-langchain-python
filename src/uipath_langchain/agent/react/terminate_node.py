"""Termination node for the Agent graph."""

from __future__ import annotations

from typing import Any, NoReturn

from langchain_core.messages import AIMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.core.chat import UiPathConversationMessageData
from uipath.runtime.errors import UiPathErrorCategory

from ...runtime.messages import UiPathChatMessagesMapper
from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from .types import AgentGraphState


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
    raise AgentRuntimeError(
        code=AgentRuntimeErrorCode.TERMINATION_LLM_RAISED_ERROR,
        title=error_message,
        detail=detail,
        category=UiPathErrorCategory.USER,
    )


def _handle_end_conversational(
    state: AgentGraphState, response_schema: type[BaseModel] | None
) -> dict[str, Any]:
    """Handle conversational agent termination by returning converted messages."""
    if state.inner_state.initial_message_count is None:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.STATE_ERROR,
            title="No initial message count in state for conversational agent execution.",
            detail="Initial message count must be set in inner_state for conversational agent execution.",
            category=UiPathErrorCategory.SYSTEM,
        )

    if response_schema is None:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.STATE_ERROR,
            title="No response schema for conversational agent termination.",
            detail="Response schema must be provided for termination of conversational agent execution.",
            category=UiPathErrorCategory.SYSTEM,
        )

    initial_count = state.inner_state.initial_message_count
    new_messages = state.messages[initial_count:]

    converted_messages: list[UiPathConversationMessageData] = []

    # For the agent-output messages, don't include tool-results. Just include agent's LLM outputs and tool-calls + inputs.
    # This is primarily since evaluations don't check for tool-results; this output represents the agent's actual choices rather than tool-results.
    if new_messages:
        converted_messages = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages=new_messages, include_tool_results=False
            )
        )

    output = {
        "uipath__agent_response_messages": [
            msg.model_dump(by_alias=True) for msg in converted_messages
        ]
    }
    validated = response_schema.model_validate(output)

    # Dump with exclude_none to prevent UiPathConversation... fields with None values from being outputted (e.g. UiPathConversationContentPartData.isTranscript).
    # May need to revisit if other output fields are added for conversational agents, where we want nulls outputted.
    return validated.model_dump(by_alias=True, exclude_none=True)


def create_terminate_node(
    response_schema: type[BaseModel] | None = None,
    is_conversational: bool = False,
):
    """Handles Agent Graph termination for multiple sources and output or error propagation to Orchestrator.

    Termination scenarios:
    1. LLM-initiated termination (END_EXECUTION_TOOL)
    2. LLM-initiated error (RAISE_ERROR_TOOL)
    3. End of conversational loop
    """

    def terminate_node(state: AgentGraphState):
        if is_conversational:
            return _handle_end_conversational(state, response_schema)
        else:
            last_message = state.messages[-1]
            if not isinstance(last_message, AIMessage):
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.ROUTING_ERROR,
                    title=f"Expected last message to be AIMessage, got {type(last_message).__name__}.",
                    detail="The terminate node requires the last message to be an AIMessage with control flow tool calls.",
                    category=UiPathErrorCategory.SYSTEM,
                )

            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]

                if tool_name == END_EXECUTION_TOOL.name:
                    return _handle_end_execution(tool_call["args"], response_schema)

                if tool_name == RAISE_ERROR_TOOL.name:
                    _handle_raise_error(tool_call["args"])

            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.ROUTING_ERROR,
                title="No control flow tool call found in terminate node.",
                detail="The terminate node was reached but no end_execution or raise_error tool call was found.",
                category=UiPathErrorCategory.SYSTEM,
            )

    return terminate_node
