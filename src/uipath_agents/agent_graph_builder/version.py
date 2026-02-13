"""Version compatibility checks for agent graph features."""

from packaging.version import InvalidVersion, Version

_PARALLEL_TOOL_CALLS_MIN_VERSION = Version("1.1.0")


def supports_openai_parallel_tool_calls(version: str) -> bool:
    """Check if the agent version supports parallel tool calls."""
    try:
        return Version(version) >= _PARALLEL_TOOL_CALLS_MIN_VERSION
    except InvalidVersion:
        return True
