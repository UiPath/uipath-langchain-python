"""Map normalized LLM-client errors into agent runtime errors.

The LLM client (uipath-llm-client / uipath-langchain-client) surfaces two shapes:
a ``UiPathError`` carrying a semantic ``error_code`` (handled by
``raise_for_llm_client_error``), and a ``UiPathAPIError`` carrying an HTTP
``status_code`` + ``body`` for provider passthrough failures (handled by
``raise_for_provider_http_error``). Both are mapped to ``AgentRuntimeError`` so
upstream handling can categorise without provider-specific logic.
"""

from typing import NoReturn

from uipath.llm_client import UiPathAPIError, UiPathError, UiPathLLMErrorCode
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

_LICENSE_ERROR_CODE = 10000
_LICENSE_TITLE = "license not available"


def raise_for_llm_client_error(error: UiPathError) -> None:
    """Raise a structured agent error for known LLM-client error codes."""
    if error.error_code == UiPathLLMErrorCode.UNSUPPORTED_MIME_TYPE:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.FILE_ERROR,
            title="Unsupported file attachment format.",
            detail=(
                "The model does not support this attachment's file type. "
                "Remove the attachment or convert it to a supported format."
                + (f" Provider detail: {error.detail}" if error.detail else "")
            ),
            category=UiPathErrorCategory.USER,
        ) from error


def _is_license_error(body: object) -> bool:
    """True only for the LLM gateway's own licensing ProblemDetails.

    Anything else -- a passthrough body, an HTML edge page, an empty body, or a
    403 carrying a different ``errorCode`` such as 10900 (authorization) -- is
    not a licensing failure, whatever its status code.
    """
    if not isinstance(body, dict):
        return False

    error_code = body.get("errorCode")
    # bool is an int subclass, so True would otherwise compare equal to 1.
    if isinstance(error_code, (int, str)) and not isinstance(error_code, bool):
        try:
            if int(error_code) == _LICENSE_ERROR_CODE:
                return True
        except ValueError:
            pass

    title = body.get("title")
    return isinstance(title, str) and title.strip().lower() == _LICENSE_TITLE


def _classify(
    status_code: int, body: object
) -> tuple[AgentRuntimeErrorCode, UiPathErrorCategory, str]:
    """Map an LLM provider HTTP status onto (code, category, title).

    403 is the only status whose meaning depends on the body; keeping the code,
    category and title decided in one place stops them drifting apart.
    """
    if status_code == 403:
        if _is_license_error(body):
            title = body.get("title") if isinstance(body, dict) else None
            return (
                AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE,
                UiPathErrorCategory.DEPLOYMENT,
                title
                if isinstance(title, str) and title.strip()
                else "License not available",
            )
        return (
            AgentRuntimeErrorCode.LLM_PROVIDER_FORBIDDEN,
            UiPathErrorCategory.DEPLOYMENT,
            "LLM provider returned HTTP 403",
        )

    title = f"LLM provider returned HTTP {status_code}"
    if status_code >= 500:
        return AgentRuntimeErrorCode.HTTP_ERROR, UiPathErrorCategory.SYSTEM, title
    return AgentRuntimeErrorCode.HTTP_ERROR, UiPathErrorCategory.UNKNOWN, title


def raise_for_provider_http_error(error: UiPathAPIError) -> NoReturn:
    """Convert a normalized ``UiPathAPIError`` into a structured ``AgentRuntimeError``.

    Reads the HTTP status code and the gateway's ``detail`` (from ``error.body``)
    and re-raises as an ``AgentRuntimeError`` chained on the original.
    """
    status_code = error.status_code
    body = error.body
    code, category, title = _classify(status_code, body)
    detail = error.body.get("detail") if isinstance(error.body, dict) else None

    raise AgentRuntimeError(
        code=code,
        title=title,
        detail=detail or error.message or str(error),
        category=category,
        status=status_code,
    ) from error
