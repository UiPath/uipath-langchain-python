"""Helper utilities for agent tests."""

from uipath_langchain.agent.exceptions import (
    AgentRuntimeErrorCode,
    AgentStartupErrorCode,
)


def agent_runtime_code(code: AgentRuntimeErrorCode) -> str:
    """Construct full error code for AgentRuntimeError.

    Args:
        code: The error code enum value

    Returns:
        Full error code string with AGENT_RUNTIME prefix

    Example:
        >>> agent_runtime_code(AgentRuntimeErrorCode.ROUTING_ERROR)
        'AGENT_RUNTIME.ROUTING_ERROR'
    """
    return f"AGENT_RUNTIME.{code.value}"


def agent_config_code(code: AgentStartupErrorCode) -> str:
    """Construct full error code for AgentStartupError.

    Args:
        code: The error code enum value

    Returns:
        Full error code string with AGENT_STARTUP prefix

    Example:
        >>> agent_config_code(AgentStartupErrorCode.LLM_INVALID_MODEL)
        'AGENT_STARTUP.LLM_INVALID_MODEL'
    """
    return f"AGENT_STARTUP.{code.value}"
