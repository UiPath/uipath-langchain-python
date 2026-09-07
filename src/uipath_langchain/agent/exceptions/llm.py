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

# A canned, provider-free replacement for the useless HTTP reason phrase. The
# relayed provider message is deliberately NOT read out of the body (PC-5002):
# it may carry customer PII, and it is already recorded on the LLM call span,
# which is tenant-scoped. It has to stand on its own -- USER is not in
# _SHOULD_WRAP_CATEGORIES, so nothing else is prepended to it.
_BAD_REQUEST_DETAIL = (
    "The model provider rejected the request as invalid. Review the agent's model "
    "settings (output-token limit, temperature, effort). The provider's own message "
    "is recorded on the LLM call span for this run."
)


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
) -> tuple[AgentRuntimeErrorCode, UiPathErrorCategory, str, str | None]:
    """Map an LLM provider HTTP status onto (code, category, title, fallback_detail).

    Only 400 and 403 are classified beyond the 5xx/other split. 404 is
    deliberately left in UNKNOWN: every 404 observed in prod over 30 days was a
    missing or unreachable model deployment (BYO relay not connected, Azure
    DeploymentNotFound, a retired Bedrock model), which is Deployment rather
    than User -- so it needs its own decision, not this one.

    403 is the only status whose meaning depends on the body; keeping the code,
    category, title and fallback detail decided in one place stops them drifting
    apart.

    ``fallback_detail`` is the customer-facing text to use when the gateway
    supplied no ProblemDetails ``detail`` of its own. ``None`` means "fall back
    to the HTTP reason phrase" -- the two-word message that PC-5002 is about, so
    only statuses whose cause we cannot name are left with it.
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
                None,
            )
        return (
            AgentRuntimeErrorCode.LLM_PROVIDER_FORBIDDEN,
            UiPathErrorCategory.DEPLOYMENT,
            "LLM provider returned HTTP 403",
            None,
        )

    if status_code == 400:
        return (
            AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST,
            UiPathErrorCategory.USER,
            "LLM provider rejected the request",
            _BAD_REQUEST_DETAIL,
        )

    title = f"LLM provider returned HTTP {status_code}"
    if status_code >= 500:
        return (
            AgentRuntimeErrorCode.HTTP_ERROR,
            UiPathErrorCategory.SYSTEM,
            title,
            None,
        )
    return (
        AgentRuntimeErrorCode.HTTP_ERROR,
        UiPathErrorCategory.UNKNOWN,
        title,
        None,
    )


def raise_for_provider_http_error(error: UiPathAPIError) -> NoReturn:
    """Convert a normalized ``UiPathAPIError`` into a structured ``AgentRuntimeError``.

    Reads the HTTP status code and the gateway's ``detail`` (from ``error.body``)
    and re-raises as an ``AgentRuntimeError`` chained on the original.
    """
    status_code = error.status_code
    body = error.body
    code, category, title, fallback_detail = _classify(status_code, body)
    gateway_detail = body.get("detail") if isinstance(body, dict) else None

    raise AgentRuntimeError(
        code=code,
        title=title,
        detail=gateway_detail or fallback_detail or error.message or str(error),
        category=category,
        status=status_code,
    ) from error
