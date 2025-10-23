"""Tool-related utility functions."""

import re


def sanitize_tool_name(name: str) -> str:
    """Sanitize tool name for LLM compatibility (alphanumeric, underscore, hyphen only, max 64 chars)."""
    trim_whitespaces = "_".join(name.split())
    sanitized_tool_name = re.sub(r"[^a-zA-Z0-9_-]", "", trim_whitespaces)
    sanitized_tool_name = sanitized_tool_name[:64]
    return sanitized_tool_name
