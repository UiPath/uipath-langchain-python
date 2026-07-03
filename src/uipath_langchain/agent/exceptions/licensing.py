"""Convert LLM provider HTTP errors into structured AgentRuntimeErrors.

Provider exceptions are first normalized to a common ``ProviderError`` (status_code + detail).
This module maps that status code to an AgentRuntimeError so
upstream handling (exception mapper, CAS bridge) can categorise by status code
without provider-specific logic.
"""

from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.chat.provider_errors import extract_provider_error

# Maps known LLM Gateway status codes to specific error codes.
# Unknown status codes fall back to HTTP_ERROR.
_LLM_STATUS_CODE_MAP: dict[int, AgentRuntimeErrorCode] = {
    403: AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE,
}


def raise_for_provider_http_error(e: BaseException) -> None:
    """Re-raise provider-specific HTTP errors as a structured AgentRuntimeError.

    Extracts the HTTP status code and the gateway's ``detail``
    from any LLM provider exception and converts it to an
    AgentRuntimeError. Does nothing if no HTTP status code can be extracted.
    """
    err = extract_provider_error(e)
    if err.status_code is None:
        return

    code = _LLM_STATUS_CODE_MAP.get(err.status_code, AgentRuntimeErrorCode.HTTP_ERROR)
    category = (
        UiPathErrorCategory.DEPLOYMENT
        if err.status_code == 403
        else UiPathErrorCategory.UNKNOWN
    )

    raise AgentRuntimeError(
        code=code,
        title=f"LLM provider returned HTTP {err.status_code}",
        detail=err.detail or str(e),
        category=category,
        status=err.status_code,
    ) from e
