"""Shared utilities for guardrail middlewares."""

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


def sanitize_tool_name(name: str) -> str:
    """Sanitize tool name for LLM compatibility (alphanumeric, underscore, hyphen only, max 64 chars)."""
    trim_whitespaces = "_".join(name.split())
    sanitized_tool_name = re.sub(r"[^a-zA-Z0-9_-]", "", trim_whitespaces)
    sanitized_tool_name = sanitized_tool_name[:64]
    return sanitized_tool_name


def create_modified_tool_request(
    request: ToolCallRequest,
    modified_args: dict[str, Any],
) -> ToolCallRequest:
    """Create a new ToolCallRequest with modified args."""
    from copy import deepcopy
    from dataclasses import replace

    modified_tool_call = deepcopy(request.tool_call)
    modified_tool_call["args"] = modified_args

    try:
        return replace(request, tool_call=modified_tool_call)
    except (TypeError, AttributeError):
        return ToolCallRequest(
            tool_call=modified_tool_call,
            tool=request.tool,
            state=request.state,
            runtime=request.runtime,
        )


def create_modified_tool_result(
    result: ToolMessage | Command[Any],
    modified_output: dict[str, Any] | str,
) -> ToolMessage | Command[Any]:
    """Create a new ToolMessage or Command with modified output."""
    import json
    from copy import deepcopy

    original_content = None
    if isinstance(result, Command):
        update = result.update if hasattr(result, "update") else {}
        messages = update.get("messages", []) if isinstance(update, dict) else []
        if messages and isinstance(messages[0], ToolMessage):
            original_content = messages[0].content
    elif isinstance(result, ToolMessage):
        original_content = result.content

    if isinstance(modified_output, dict):
        if (
            isinstance(original_content, str)
            and len(modified_output) == 1
            and "output" in modified_output
            and isinstance(modified_output["output"], str)
        ):
            formatted_output = modified_output["output"]
        else:
            formatted_output = json.dumps(modified_output)
    else:
        formatted_output = modified_output

    if isinstance(result, Command):
        update = result.update if hasattr(result, "update") else {}
        messages = update.get("messages", []) if isinstance(update, dict) else []
        if messages and isinstance(messages[0], ToolMessage):
            modified_message = deepcopy(messages[0])
            modified_message.content = formatted_output
            new_update: dict[str, Any] = (
                dict(deepcopy(update)) if isinstance(update, dict) else {}
            )
            new_update["messages"] = [modified_message]
            return Command(update=new_update)
        return result
    elif isinstance(result, ToolMessage):
        modified_message = deepcopy(result)
        modified_message.content = formatted_output
        return modified_message
    else:
        return result


def extract_text_from_messages(messages: list[BaseMessage]) -> str:
    """Extract text content from messages."""
    text_parts = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
    return "\n".join(text_parts)
