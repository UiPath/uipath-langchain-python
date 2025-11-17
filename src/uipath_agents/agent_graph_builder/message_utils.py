"""Message creation and template interpolation utilities."""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, List, Pattern

from langchain_core.messages import HumanMessage, SystemMessage
from uipath.agent.models.agent import AgentMessage
from uipath.agent.react import AGENT_SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Whitelist pattern: only allow alphanumeric, dots, and underscores in arguments syntax
SAFE_ARGUMENT_PATTERN: Pattern[str] = re.compile(r"^[a-zA-Z0-9_.]+$")


def apply_system_prompt_template(system_prompt_content: str, agent_name: str) -> str:
    """Apply AGENT_SYSTEM_PROMPT_TEMPLATE with systemPrompt, currentDate, and agentName."""
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return (
        AGENT_SYSTEM_PROMPT_TEMPLATE.replace("{{systemPrompt}}", system_prompt_content)
        .replace("{{currentDate}}", current_date)
        .replace("{{agentName}}", agent_name)
    )


def build_agent_messages(
    agent_messages: List[AgentMessage],
    input_arguments: dict[str, Any],
    agent_name: str,
) -> list[SystemMessage | HumanMessage]:
    """Convert agent messages to LangChain messages with {{variable}} interpolation.

    System messages are wrapped in AGENT_SYSTEM_PROMPT_TEMPLATE with currentDate and agentName.
    Expects exactly one system message and one user message.

    Raises:
        ValueError: If system or user message is missing
    """
    system_message = next((msg for msg in agent_messages if msg.role == "system"), None)
    if system_message is None:
        raise ValueError("Agent configuration must contain exactly one system message")

    user_message = next((msg for msg in agent_messages if msg.role == "user"), None)
    if user_message is None:
        raise ValueError("Agent configuration must contain exactly one user message")

    system_prompt_content = interpolate_message(
        system_message.content or "", input_arguments
    )
    system_prompt = apply_system_prompt_template(system_prompt_content, agent_name)
    user_prompt = interpolate_message(user_message.content or "", input_arguments)

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def safe_get_nested(data: dict[str, Any], path: str) -> Any:
    """Get nested dictionary value using dot notation (e.g., "user.email")."""
    keys = path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def serialize_argument(
    value: str | int | float | bool | list[Any] | dict[str, Any] | None,
) -> str:
    """Serialize value for interpolation: primitives as-is, collections as JSON."""
    if value is None:
        return ""
    if isinstance(value, (list, dict, bool)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def interpolate_message(content: str, input_values: dict[str, Any]) -> str:
    """Replace {{variable}} placeholders with values. Supports nested paths (e.g., {{user.email}}).

    Uses whitelist pattern to prevent injection attacks.
    """
    argument_placeholder_pattern = r"\{\{([^}]+)\}\}"
    matches = re.findall(argument_placeholder_pattern, content)

    interpolated = content
    for field_path in matches:
        field_path = field_path.strip()

        # Validate field path to prevent injection
        if not SAFE_ARGUMENT_PATTERN.match(field_path):
            logger.warning(
                f"Skipping unsafe placeholder '{{{{{field_path}}}}}' - "
                f"only alphanumeric, dots, and underscores are allowed"
            )
            continue

        value = safe_get_nested(input_values, field_path)

        if value is not None:
            placeholder = f"{{{{{field_path}}}}}"
            interpolated = interpolated.replace(placeholder, serialize_argument(value))

    return interpolated
