"""Map normalized LLM-client errors into agent runtime errors.

The LLM client (uipath-llm-client / uipath-langchain-client) surfaces two shapes:
a ``UiPathError`` carrying a semantic ``error_code`` (handled by
``raise_for_llm_client_error``), and a ``UiPathAPIError`` carrying an HTTP
``status_code`` + ``body`` for provider passthrough failures (handled by
``raise_for_provider_http_error``). Both are mapped to ``AgentRuntimeError`` so
upstream handling can categorise without provider-specific logic.
"""

from collections.abc import Sequence
from typing import NoReturn

from uipath.llm_client import UiPathAPIError, UiPathError, UiPathLLMErrorCode
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

_LICENSE_ERROR_CODE = 10000
_LICENSE_TITLE = "license not available"

_BAD_REQUEST_DETAIL = (
    "The model provider rejected the request as invalid. Review the agent's model "
    "settings (output-token limit, temperature, effort). The provider's own message "
    "is recorded on the LLM call span for this run."
)

_Verdict = tuple[AgentRuntimeErrorCode, UiPathErrorCategory, str, str | None]

_NOT_FOUND_SIGNATURES: tuple[tuple[tuple[str, ...], _Verdict], ...] = (
    (
        ("no active connection found for endpoint",),
        (
            AgentRuntimeErrorCode.LLM_BYO_CONNECTION_UNAVAILABLE,
            UiPathErrorCategory.DEPLOYMENT,
            "The agent's model connection is not available",
            "The model this agent uses is served through a bring-your-own-model "
            "connection whose relay is not connected. Start the relay client for "
            "that connection, or reload the relay on your nodes if you recently "
            "changed its configuration, then run the agent again.",
        ),
    ),
    (
        ("deploymentnotfound",),
        (
            AgentRuntimeErrorCode.LLM_PROVIDER_NOT_FOUND,
            UiPathErrorCategory.DEPLOYMENT,
            "The agent's model deployment does not exist",
            "If you are using a Bring Your Own configuration "
            "make sure it is correctly configured. If the error "
            "persists, contact your administrator.",
        ),
    ),
    (
        ("reached the end of its life",),
        (
            AgentRuntimeErrorCode.LLM_PROVIDER_NOT_FOUND,
            UiPathErrorCategory.DEPLOYMENT,
            "The agent's model has been retired",
            "The provider has retired the model version this agent is configured "
            "to use. Point the agent at a currently supported model.",
        ),
    ),
)


def raise_for_llm_client_error(
    error: UiPathError, attachments: Sequence[str] = ()
) -> None:
    """Raise a structured agent error for known LLM-client error codes.

    Args:
        error: The LLM-client error.
        attachments: Descriptions of the files sent with the request (see
            :func:`describe_attachment`), named in the error when given.
    """
    if error.error_code == UiPathLLMErrorCode.UNSUPPORTED_MIME_TYPE:
        subject = (
            f"one of these attachments: {', '.join(attachments)}"
            if attachments
            else "this attachment"
        )
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.FILE_ERROR,
            title="Unsupported file attachment format.",
            detail=(
                f"The model does not support the file type of {subject}. "
                "Remove the attachment or convert it to a supported format."
                + (f" Provider detail: {error.detail}" if error.detail else "")
            ),
            category=UiPathErrorCategory.USER,
        ) from error


def describe_attachment(name: str, mime_type: str) -> str:
    """An attachment as it is named in user-facing errors: ``'a.csv' (text/csv)``."""
    return f"'{name}' ({mime_type or 'no file type'})"


# Lowercased fragments of the providers' "input exceeds the context window" 400s.
_CONTEXT_LIMIT_MARKERS = (
    "prompt is too long",
    "input is too long",
    "context window",
    "context length",
    "maximum context",
    "too many tokens",
    # Gemini: "The input token count (N) exceeds the maximum number of tokens
    # allowed (M)."
    "maximum number of tokens",
)


def raise_for_file_analysis_http_error(
    error: UiPathAPIError, attachments: Sequence[str]
) -> NoReturn:
    """Convert a provider rejection of a file-analysis request into an agent error.

    A 400 on a request whose only inputs are a task and the attached files is
    about those files: either the model cannot read them, or their content does
    not fit its context window. The error names the files and says which, so
    the user knows what to change -- unlike the generic bad-request detail,
    which points at model settings. Any other status, and a 400 carrying the
    gateway's own ProblemDetails ``detail`` (the gateway, not the provider,
    rejected the call), is mapped as usual by
    :func:`raise_for_provider_http_error`.

    The provider's own message is matched for the context-window case but never
    copied into the error: it may carry customer content, and it is already
    recorded on the LLM call span.

    Args:
        error: The normalized provider error.
        attachments: Descriptions of the files sent with the request (see
            :func:`describe_attachment`).
    """
    gateway_detail = error.body.get("detail") if isinstance(error.body, dict) else None
    if error.status_code != 400 or (
        isinstance(gateway_detail, str) and gateway_detail.strip()
    ):
        raise_for_provider_http_error(error)

    files = ", ".join(attachments) or "the attached files"
    fields = _body_fields(error.body)
    if any(marker in field for field in fields for marker in _CONTEXT_LIMIT_MARKERS):
        title = "The attached files are too large for the model"
        detail = (
            f"The content of {files} does not fit in the model's context window. "
            "Split the file or reduce its content, or use a model with a larger "
            "context window."
        )
    else:
        title = "The model could not read the attached files"
        detail = (
            f"The model provider rejected the request to analyze {files}. The "
            "model may not support this file type or its content. Convert the "
            "file to PDF or use a different model, then run the agent again."
        )

    raise AgentRuntimeError(
        code=AgentRuntimeErrorCode.FILE_ERROR,
        title=title,
        detail=detail,
        category=UiPathErrorCategory.USER,
        status=error.status_code,
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


def _body_fields(body: object) -> list[str]:
    """The body's free-text fields, lowercased, for marker matching only."""
    if isinstance(body, str):
        return [body.lower()]
    if not isinstance(body, dict):
        return []

    sources: list[object] = [body]
    error = body.get("error")
    if isinstance(error, dict):
        sources.append(error)
    elif isinstance(error, str):
        sources.append({"message": error})

    return [
        value.lower()
        for source in sources
        if isinstance(source, dict)
        for key in ("message", "code", "detail", "title")
        if isinstance(value := source.get(key), str)
    ]


def _match_not_found_signature(body: object) -> _Verdict | None:
    """The verdict for a 404 whose body names its own cause, else ``None``."""
    fields = _body_fields(body)
    for markers, verdict in _NOT_FOUND_SIGNATURES:
        if any(all(marker in field for marker in markers) for field in fields):
            return verdict
    return None


def _forbidden_verdict(body: object) -> _Verdict:
    """The verdict for a 403, whose meaning is in the body rather than the status."""
    if not _is_license_error(body):
        return (
            AgentRuntimeErrorCode.LLM_PROVIDER_FORBIDDEN,
            UiPathErrorCategory.DEPLOYMENT,
            "LLM provider returned HTTP 403",
            None,
        )

    title = body.get("title") if isinstance(body, dict) else None
    return (
        AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE,
        UiPathErrorCategory.DEPLOYMENT,
        title if isinstance(title, str) and title.strip() else "License not available",
        None,
    )


def _status_verdict(status_code: int, body: object) -> _Verdict:
    """The verdict this mapping decides for a status, before the gateway's detail.

    Only 400, 403 and 404 are classified beyond the 5xx/other split.

    403 and 404 are the statuses whose meaning depends on the body; keeping the
    code, category, title and detail decided in one place stops them drifting
    apart. Both name a cause only for a body that names its own --
    ``_is_license_error`` for 403, ``_NOT_FOUND_SIGNATURES`` for 404 -- and
    leave the rest unnamed rather than guessing.
    """
    if status_code == 403:
        return _forbidden_verdict(body)

    if status_code == 400:
        # The relayed provider message is deliberately not read out of the
        # body: it may carry customer PII, and it is already recorded on the
        # LLM call span, which is tenant-scoped.
        return (
            AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST,
            UiPathErrorCategory.USER,
            "LLM provider rejected the request",
            _BAD_REQUEST_DETAIL,
        )

    if status_code == 404 and (verdict := _match_not_found_signature(body)) is not None:
        return verdict

    return (
        AgentRuntimeErrorCode.HTTP_ERROR,
        UiPathErrorCategory.SYSTEM
        if status_code >= 500
        else UiPathErrorCategory.UNKNOWN,
        f"LLM provider returned HTTP {status_code}",
        None,
    )


def _classify(status_code: int, body: object) -> _Verdict:
    """Map an LLM provider HTTP status onto (code, category, title, detail).

    The gateway's own ProblemDetails ``detail`` is first-party UiPath text and
    more specific, so it wins over the detail ``_status_verdict`` decided. A
    ``detail`` of ``None`` means "fall back to the HTTP reason phrase" -- the
    useless two-word message, so only statuses whose cause neither the gateway
    nor this mapping can name are left with it.
    """
    code, category, title, own_detail = _status_verdict(status_code, body)
    gateway_detail = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(gateway_detail, str):
        gateway_detail = None
    return code, category, title, gateway_detail or own_detail


def raise_for_provider_http_error(error: UiPathAPIError) -> NoReturn:
    """Convert a normalized ``UiPathAPIError`` into a structured ``AgentRuntimeError``.

    Reads the HTTP status code and ``error.body``, and re-raises as an
    ``AgentRuntimeError`` chained on the original. When ``_classify`` names no
    detail, the error's own message -- the HTTP reason phrase -- is all that is
    left.
    """
    status_code = error.status_code
    code, category, title, detail = _classify(status_code, error.body)

    raise AgentRuntimeError(
        code=code,
        title=title,
        detail=detail or error.message or str(error),
        category=category,
        status=status_code,
    ) from error
