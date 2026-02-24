"""Maps generic exceptions into structured UiPath runtime errors.

Classifies generic Python exceptions into AgentRuntimeError or AgentStartupError
with structured error codes, titles, and categories. Existing UiPathBaseRuntimeError
instances pass through unchanged.
"""

from typing import TypeVar

import httpx
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import (
    UiPathBaseRuntimeError,
    UiPathErrorCategory,
)
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)

C = TypeVar("C", AgentRuntimeErrorCode, AgentStartupErrorCode)


class ExceptionMapper:
    """Maps generic exceptions into AgentRuntimeError or AgentStartupError.

    Provides context-aware translation:
    - map_runtime(): For errors during graph execution
    - map_config(): For errors during initialization/setup

    Existing UiPathBaseRuntimeError instances are returned as-is.
    """

    @staticmethod
    def map_runtime(e: BaseException) -> UiPathBaseRuntimeError:
        """Maps runtime exceptions (during graph execution).

        Args:
            e: The exception to map.

        Returns:
            AgentRuntimeError with user-actionable fields.
            If e is already a UiPathBaseRuntimeError, returns it unchanged.
        """
        if isinstance(e, UiPathBaseRuntimeError):
            return e

        code, title, detail, category, status = _classify_runtime(e)

        return AgentRuntimeError(
            code=code,
            title=title,
            detail=detail,
            category=category,
            status=status,
            include_traceback=False,
        )

    @staticmethod
    def map_config(e: BaseException) -> UiPathBaseRuntimeError:
        """Maps config exceptions (during initialization/setup).

        Args:
            e: The exception to map.

        Returns:
            AgentStartupError with user-actionable fields.
            If e is already a UiPathBaseRuntimeError, returns it unchanged.
        """
        if isinstance(e, UiPathBaseRuntimeError):
            return e

        code, title, detail, category, status = _classify_config(e)

        return AgentStartupError(
            code=code,
            title=title,
            detail=detail,
            category=category,
            status=status,
            include_traceback=False,
        )


# --- Classification helpers ---


def _classify_runtime(
    e: BaseException,
) -> tuple[AgentRuntimeErrorCode, str, str, UiPathErrorCategory, int | None]:
    """Classify runtime exceptions into error code, title, detail, category, and status."""
    if isinstance(e, EnrichedException):
        return _classify_http_error(e, e.status_code, AgentRuntimeErrorCode)
    if isinstance(e, httpx.HTTPStatusError):
        return _classify_http_error(e, e.response.status_code, AgentRuntimeErrorCode)

    return (
        AgentRuntimeErrorCode.UNEXPECTED_ERROR,
        f"Unexpected error: {type(e).__name__}",
        str(e),
        UiPathErrorCategory.UNKNOWN,
        None,
    )


def _classify_config(
    e: BaseException,
) -> tuple[AgentStartupErrorCode, str, str, UiPathErrorCategory, int | None]:
    """Classify config exceptions into error code, title, detail, category, and status."""
    if isinstance(e, EnrichedException):
        return _classify_http_error(e, e.status_code, AgentStartupErrorCode)
    if isinstance(e, httpx.HTTPStatusError):
        return _classify_http_error(e, e.response.status_code, AgentStartupErrorCode)

    return (
        AgentStartupErrorCode.UNEXPECTED_ERROR,
        f"Unexpected error {type(e).__name__}",
        str(e),
        UiPathErrorCategory.UNKNOWN,
        None,
    )


def _classify_http_error(
    e: BaseException,
    status_code: int | str | None,
    code_type: type[C],
) -> tuple[C, str, str, UiPathErrorCategory, int | None]:
    """Map HTTP status codes to error classification."""
    sc = int(status_code) if status_code is not None else 0

    return (
        code_type.HTTP_ERROR,
        "HTTP Request Failed",
        str(e),
        UiPathErrorCategory.UNKNOWN,
        sc if sc else None,
    )
