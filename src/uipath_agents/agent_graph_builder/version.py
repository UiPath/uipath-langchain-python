"""Version compatibility checks for agent graph features."""

from packaging.version import InvalidVersion, Version

_PARALLEL_TOOL_CALLS_MIN_VERSION = Version("1.1.0")


def supports_parallel_tool_calls(version: str, model_name: str) -> bool:
    """Check if the agent version and model name supports parallel tool calls."""
    try:
        return (
            Version(version) >= _PARALLEL_TOOL_CALLS_MIN_VERSION
            or "gpt" not in model_name.lower()
        )
    except InvalidVersion:
        return True
