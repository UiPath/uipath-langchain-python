"""ReAct Agent loop utilities."""

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model


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


def add_job_attachments(
    left: dict[str, Attachment], right: dict[str, Attachment]
) -> dict[str, Attachment]:
    """Merge attachment dictionaries, with right values taking precedence.

    This reducer function merges two dictionaries of attachments by UUID string.
    If the same UUID exists in both dictionaries, the value from 'right' takes precedence.

    Args:
        left: Existing dictionary of attachments keyed by UUID string
        right: New dictionary of attachments to merge

    Returns:
        Merged dictionary with right values overriding left values for duplicate keys
    """
    if not right:
        return left

    if not left:
        return right

    return {**left, **right}
