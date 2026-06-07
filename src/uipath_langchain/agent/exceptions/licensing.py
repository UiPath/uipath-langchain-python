"""Convert LLM provider HTTP errors into structured AgentRuntimeErrors.

Each LLM provider wraps HTTP errors in a different exception type:
- OpenAI: openai.PermissionDeniedError  → e.status_code
- Vertex: google.genai.errors.ClientError → e.code
- Bedrock: botocore.exceptions.ClientError → e.response dict

This module extracts the HTTP status code from any of these and re-raises
as an AgentRuntimeError so that upstream error handling (exception mapper,
CAS bridge) can categorise by status code without provider-specific logic.
"""

from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)


def _extract_status_code(e: BaseException) -> int | None:
    """Extract HTTP status code from any provider-specific exception.

    Supports OpenAI (status_code), Vertex/google.genai (code), and
    Bedrock/botocore (response dict). Walks __cause__ chain to handle
    LangChain wrapper exceptions (e.g. ChatGoogleGenerativeAIError).
    """
    # OpenAI: e.status_code
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int):
        return sc

    # Vertex (google.genai.errors.APIError): e.code
    sc = getattr(e, "code", None)
    if isinstance(sc, int):
        return sc

    # Bedrock (botocore.exceptions.ClientError): e.response dict
    resp = getattr(e, "response", None)
    if isinstance(resp, dict):
        sc = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(sc, int):
            return sc

    # Walk __cause__ chain
    cause = getattr(e, "__cause__", None)
    if cause is not None and cause is not e:
        return _extract_status_code(cause)

    return None


# Maps known LLM Gateway status codes to specific error codes.
# Unknown status codes fall back to HTTP_ERROR.
_LLM_STATUS_CODE_MAP: dict[int, AgentRuntimeErrorCode] = {
    403: AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE,
}


def raise_for_provider_http_error(e: BaseException) -> None:
    """Re-raise provider-specific HTTP errors as a structured AgentRuntimeError.

    Extracts the HTTP status code from any LLM provider exception and
    converts it to an AgentRuntimeError with the status code preserved.
    Known status codes (e.g. 403) get a specific error code so upstream
    handlers can match on the suffix. Does nothing if no HTTP status code
    can be extracted.
    """
    sc = _extract_status_code(e)
    if sc is None:
        return

    code = _LLM_STATUS_CODE_MAP.get(sc, AgentRuntimeErrorCode.HTTP_ERROR)

    if sc == 403:
        category = UiPathErrorCategory.DEPLOYMENT
    else:
        category = UiPathErrorCategory.UNKNOWN

    raise AgentRuntimeError(
        code=code,
        title=f"LLM provider returned HTTP {sc}",
        detail=str(e),
        category=category,
        status=sc,
    ) from e
