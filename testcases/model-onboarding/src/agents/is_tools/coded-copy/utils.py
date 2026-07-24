import json
import re
from typing import Any, Pattern

SAFE_ARGUMENT_PATTERN: Pattern[str] = re.compile(r"^[a-zA-Z0-9_.\[\]*]+$")


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


def interpolate_legacy_message(content: str, input_values: dict[str, Any]) -> str:
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
            continue

        value = safe_get_nested(input_values, field_path)

        if value is not None:
            placeholder = f"{{{{{field_path}}}}}"
            interpolated = interpolated.replace(
                placeholder, serialize_argument(str(value))
            )

    return interpolated
