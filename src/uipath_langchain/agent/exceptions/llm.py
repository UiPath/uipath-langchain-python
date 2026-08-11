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


def _extract_provider_detail(body: object) -> str | None:
    """Pull the human-readable message out of an error response body.

    Tries the gateway's own envelope (``{"detail": ...}``) first, then the
    vendor envelopes forwarded on passthrough 4xx responses (OpenAI/Anthropic
    ``{"error": {"message": ...}}``, Vertex list-wrapped variants, flat
    ``{"message": ...}``), then falls back to a raw text body.
    """
    if isinstance(body, list) and body:
        return _extract_provider_detail(body[0])
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = body.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(body, str) and body.strip():
        return body.strip()[:2000]
    return None


def raise_for_provider_http_error(error: UiPathAPIError) -> NoReturn:
    """Convert a normalized ``UiPathAPIError`` into a structured ``AgentRuntimeError``.

    Reads the HTTP status code and the error message from ``error.body`` (the
    gateway's ``detail`` envelope or the provider's own error envelope) and
    re-raises as an ``AgentRuntimeError`` chained on the original. A 400 is the
    caller's request/configuration, so it is categorised USER and surfaced
    unwrapped; other unknown statuses keep the generic UNKNOWN wrapping.
    """
    status_code = error.status_code
    code = _LLM_STATUS_CODE_MAP.get(status_code, AgentRuntimeErrorCode.HTTP_ERROR)
    if status_code == 403:
        category = UiPathErrorCategory.DEPLOYMENT
    elif status_code == 400:
        category = UiPathErrorCategory.USER
    else:
        category = UiPathErrorCategory.UNKNOWN
    detail = _extract_provider_detail(error.body)

    raise AgentRuntimeError(
        code=code,
        title=f"LLM provider returned HTTP {status_code}",
        detail=detail or error.message or str(error),
        category=category,
        status=status_code,
    ) from error
