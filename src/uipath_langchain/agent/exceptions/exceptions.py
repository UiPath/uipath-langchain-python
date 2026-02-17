"""Exceptions for the agent graph."""

from enum import Enum

from uipath.runtime.errors import (
    UiPathBaseRuntimeError,
    UiPathErrorCategory,
)

_SHOULD_WRAP_CATEGORIES = {UiPathErrorCategory.SYSTEM, UiPathErrorCategory.UNKNOWN}


def _try_wrap(
    category: UiPathErrorCategory,
    should_wrap: bool | None,
    wrap_prefix: str,
    detail: str,
) -> str:
    if should_wrap is None:
        should_wrap = category in _SHOULD_WRAP_CATEGORIES
    if not should_wrap:
        return detail
    detail = f"{wrap_prefix}\nError Details:\n{detail}"
    return detail


class AgentRuntimeErrorCode(str, Enum):
    """Error codes for exceptions thrown within the agent graph during execution."""

    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
    HTTP_ERROR = "HTTP_ERROR"

    # Routing
    ROUTING_ERROR = "ROUTING_ERROR"
    THINKING_LIMIT_EXCEEDED = "THINKING_LIMIT_EXCEEDED"

    # Termination
    TERMINATION_MAX_ITERATIONS = "TERMINATION_MAX_ITERATIONS"
    TERMINATION_LLM_RAISED_ERROR = "TERMINATION_LLM_RAISED_ERROR"
    TERMINATION_GUARDRAIL_VIOLATION = "TERMINATION_GUARDRAIL_VIOLATION"
    TERMINATION_GUARDRAIL_ERROR = "TERMINATION_GUARDRAIL_ERROR"
    TERMINATION_ESCALATION_REJECTED = "TERMINATION_ESCALATION_REJECTED"
    TERMINATION_ESCALATION_ERROR = "TERMINATION_ESCALATION_ERROR"

    # State
    STATE_ERROR = "STATE_ERROR"

    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    TOOL_INVALID_WRAPPER_STATE = "TOOL_INVALID_WRAPPER_STATE"


class AgentStartupErrorCode(str, Enum):
    """Error codes for agent startup errors detected before graph execution."""

    # General configuration errors
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"

    # HTTP/Network errors
    HTTP_ERROR = "HTTP_ERROR"

    # Package file not found
    FILE_NOT_FOUND = "FILE_NOT_FOUND"

    # LLM configuration errors
    LLM_INVALID_MODEL = "LLM_INVALID_MODEL"


class AgentRuntimeError(UiPathBaseRuntimeError):
    """Custom exception for agent loop runtime errors with structured error information.

    This exception is raised for errors that occur during agent graph execution
    (runtime context). SYSTEM and UNKNOWN category errors are automatically
    wrapped with user-friendly messaging unless should_wrap is explicitly False.

    Args:
        code: Error code identifying the specific type of runtime error
        title: Brief error title for display
        detail: Detailed error message with context
        category: Error category (USER, DEPLOYMENT, SYSTEM, UNKNOWN)
        status: Optional HTTP status code if applicable
        should_wrap: Controls wrapping. None (default) infers from category:
            True for SYSTEM/UNKNOWN, False for USER/DEPLOYMENT.
            Explicit True/False overrides inference.
        include_traceback: Whether to include stack trace in error output.
            Defaults to False for clean user-facing messages.
    """

    _WRAP_PREFIX = "An unexpected error occurred during agent execution, please try again later or contact your Administrator."

    def __init__(
        self,
        code: AgentRuntimeErrorCode,
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.UNKNOWN,
        status: int | None = None,
        should_wrap: bool | None = None,
        include_traceback: bool = False,
    ):
        detail = _try_wrap(category, should_wrap, self._WRAP_PREFIX, detail)
        super().__init__(
            code.value,
            title,
            detail,
            category,
            status,
            include_traceback=include_traceback,
            prefix="AGENT_RUNTIME",
        )


class AgentStartupError(UiPathBaseRuntimeError):
    """Custom exception for agent startup errors with structured error information.

    This exception is raised for errors that occur during agent initialization
    and setup (config context). SYSTEM and UNKNOWN category errors are automatically
    wrapped with user-friendly messaging unless should_wrap is explicitly False.

    Args:
        code: Error code identifying the specific type of startup error
        title: Brief error title for display
        detail: Detailed error message with context
        category: Error category (USER, DEPLOYMENT, SYSTEM, UNKNOWN)
        status: Optional HTTP status code if applicable
        should_wrap: Controls wrapping. None (default) infers from category:
            True for SYSTEM/UNKNOWN, False for USER/DEPLOYMENT.
            Explicit True/False overrides inference.
        include_traceback: Whether to include stack trace in error output.
            Defaults to False for clean user-facing messages.
    """

    _WRAP_PREFIX = "An unexpected error occurred during agent startup, please try again later or contact your Administrator."

    def __init__(
        self,
        code: AgentStartupErrorCode,
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.UNKNOWN,
        status: int | None = None,
        should_wrap: bool | None = None,
        include_traceback: bool = False,
    ):
        detail = _try_wrap(category, should_wrap, self._WRAP_PREFIX, detail)
        super().__init__(
            code.value,
            title,
            detail,
            category,
            status,
            prefix="AGENT_STARTUP",
            include_traceback=include_traceback,
        )
