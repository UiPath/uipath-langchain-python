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

# Maps known LLM Gateway status codes to specific error codes.
# Unknown status codes fall back to HTTP_ERROR.
_LLM_STATUS_CODE_MAP: dict[int, AgentRuntimeErrorCode] = {
    403: AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE,
}


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


def _category_for_status(status_code: int) -> UiPathErrorCategory:
    """Map LLM provider HTTP statuses to their runtime error category."""
    if status_code == 403:
        return UiPathErrorCategory.DEPLOYMENT
    if status_code >= 500:
        return UiPathErrorCategory.SYSTEM
    return UiPathErrorCategory.UNKNOWN


def raise_for_provider_http_error(error: UiPathAPIError) -> NoReturn:
    """Convert a normalized ``UiPathAPIError`` into a structured ``AgentRuntimeError``.

    Reads the HTTP status code and the gateway's ``detail`` (from ``error.body``) and
    re-raises. Only the gateway ``detail`` is surfaced — the vendor's passthrough
    message can echo request content, so it stays on the trace span (excluded from
    App Insights), never the run record. ``from None`` keeps the chained
    ``UiPathAPIError`` string (which embeds the raw body) out of ``format_exc()`` too.
    """
    status_code = error.status_code
    code = _LLM_STATUS_CODE_MAP.get(status_code, AgentRuntimeErrorCode.HTTP_ERROR)
    category = _category_for_status(status_code)
    detail = error.body.get("detail") if isinstance(error.body, dict) else None

    raise AgentRuntimeError(
        code=code,
        title=f"LLM provider returned HTTP {status_code}",
        detail=detail or error.message,
        category=category,
        status=status_code,
    ) from None
