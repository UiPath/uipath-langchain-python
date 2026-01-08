"""ReAct Agent loop utilities."""

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState


def resolve_input_model(
    input_schema: dict[str, Any] | None,
) -> type[BaseModel]:
    """Resolve the input model from the input schema."""
    if input_schema:
        return create_model(input_schema)

    return BaseModel


def resolve_output_model(
    output_schema: dict[str, Any] | None,
) -> type[BaseModel]:
    """Fallback to default end_execution tool schema when no agent output schema is provided."""
    if output_schema:
        return create_model(output_schema)

    return END_EXECUTION_TOOL.args_schema


def count_consecutive_thinking_messages(messages: Sequence[BaseMessage]) -> int:
    """Count consecutive AIMessages without tool calls at end of message history."""
    if not messages:
        return 0

    count = 0
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            break

        if message.tool_calls:
            break

        if not message.content:
            break

        count += 1

    return count


def extract_tool_call_from_state(
    state: AgentGraphState,
    tool_name: str,
    tool_call_id: str | None = None,
    return_message: bool = False,
) -> ToolCall | None | tuple[ToolCall | None, AIMessage | None]:
    """
    Extract tool call from state using consistent logic.

    Search order:
    1. If tool_call_id is provided, search for tool call with matching id and name
    2. Otherwise, find first tool call with matching name from the last AI message

    Args:
        state: The agent graph state
        tool_name: Name of the tool to find
        tool_call_id: Optional tool call id to search for
        return_message: If True, returns tuple of (tool_call, message) instead of just tool_call

    Returns:
        The matching ToolCall if found, None otherwise. If return_message is True,
        returns tuple of (ToolCall | None, AIMessage | None).
    """
    if not state.messages:
        return (None, None) if return_message else None

    # 1. If tool_call_id is provided, search for tool call with matching id and name
    if tool_call_id is not None:
        for message in reversed(state.messages):
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    if (
                        tool_call["id"] == tool_call_id
                        and tool_call["name"] == tool_name
                    ):
                        return (tool_call, message) if return_message else tool_call
        return (None, None) if return_message else None

    # 2. Find first tool call with matching name from the last AI message
    for message in reversed(state.messages):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call["name"] == tool_name:
                    return (tool_call, message) if return_message else tool_call
            break

    return (None, None) if return_message else None
