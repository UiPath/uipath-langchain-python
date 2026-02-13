"""Tool-related utility functions."""

import re
from typing import Any

from uipath.agent.models.agent import TaskTitle, TextBuilderTaskTitle
from uipath.agent.utils.text_tokens import build_string_from_tokens


def sanitize_tool_name(name: str) -> str:
    """Sanitize tool name for LLM compatibility (alphanumeric, underscore, hyphen only, max 64 chars)."""
    trim_whitespaces = "_".join(name.split())
    sanitized_tool_name = re.sub(r"[^a-zA-Z0-9_-]", "", trim_whitespaces)
    sanitized_tool_name = sanitized_tool_name[:64]
    return sanitized_tool_name


def sanitize_dict_for_serialization(args: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic models in args to dicts."""
    converted_args: dict[str, Any] = {}
    for key, value in args.items():
        # handle Pydantic model
        if hasattr(value, "model_dump"):
            converted_args[key] = value.model_dump()

        elif isinstance(value, list):
            # handle list of Pydantic models
            converted_list = []
            for item in value:
                if hasattr(item, "model_dump"):
                    converted_list.append(item.model_dump())
                elif hasattr(item, "value"):
                    converted_list.append(item.value)
                else:
                    converted_list.append(item)
            converted_args[key] = converted_list

        # handle enum-like objects with value attribute
        elif hasattr(value, "value"):
            converted_args[key] = value.value

        # handle regular value or unexpected type
        else:
            converted_args[key] = value
    return converted_args


def resolve_task_title(
    task_title: TaskTitle | str | None,
    agent_input: dict[str, Any],
    default_title: str = "Escalation Task",
) -> str:
    """Resolve task title based on channel configuration."""
    if isinstance(task_title, TextBuilderTaskTitle):
        return build_string_from_tokens(task_title.tokens, agent_input)

    if isinstance(task_title, str):
        return task_title

    return default_title
