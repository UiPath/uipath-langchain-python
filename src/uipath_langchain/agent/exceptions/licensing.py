"""Map a normalized LLM-client ``UiPathAPIError`` into an ``AgentRuntimeError``.

The LLM client (uipath-llm-client / uipath-langchain-client) normalizes provider
HTTP errors into a ``UiPathAPIError`` carrying ``status_code`` and ``body``. This
module maps that status code to an ``AgentRuntimeError`` so upstream handling
(exception mapper, CAS bridge) can categorise without provider-specific logic.
"""

from typing import NoReturn

from uipath.llm_client import UiPathAPIError
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


def raise_for_provider_http_error(error: UiPathAPIError) -> NoReturn:
    """Convert a normalized ``UiPathAPIError`` into a structured ``AgentRuntimeError``.

    Reads the HTTP status code and the gateway's ``detail`` (from ``error.body``)
    and re-raises as an ``AgentRuntimeError`` chained on the original.
    """
    status_code = error.status_code
    code = _LLM_STATUS_CODE_MAP.get(status_code, AgentRuntimeErrorCode.HTTP_ERROR)
    category = (
        UiPathErrorCategory.DEPLOYMENT
        if status_code == 403
        else UiPathErrorCategory.UNKNOWN
    )
    detail = error.body.get("detail") if isinstance(error.body, dict) else None

    raise AgentRuntimeError(
        code=code,
        title=f"LLM provider returned HTTP {status_code}",
        detail=detail or error.message or str(error),
        category=category,
        status=status_code,
    ) from error
