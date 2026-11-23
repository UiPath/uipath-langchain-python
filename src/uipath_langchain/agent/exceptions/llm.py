"""Map normalized LLM-client errors into agent runtime errors.

The LLM client (uipath-llm-client / uipath-langchain-client) surfaces two shapes:
a ``UiPathError`` carrying a semantic ``error_code`` (handled by
``raise_for_llm_client_error``), and a ``UiPathAPIError`` carrying an HTTP
``status_code`` + ``body`` for provider passthrough failures (handled by
``raise_for_provider_http_error``). Both are mapped to ``AgentRuntimeError`` so
upstream handling can categorise without provider-specific logic. A request the
LLM gateway's centralized (AI Trust Layer) guardrails blocked is recognised
first, by the gateway's error code (``centralized_guardrail_error``).
"""

import re
from collections.abc import Sequence
from typing import NoReturn

from uipath.llm_client import UiPathAPIError, UiPathError, UiPathLLMErrorCode
from uipath.llm_client.utils.exceptions import as_uipath_error
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

# The LLM client has already retried a 429 (with backoff, honouring Retry-After)
# before this mapping sees it, so "try again" alone is not enough advice.
_RATE_LIMIT_DETAIL = (
    "The model call was refused as too many requests (HTTP 429) and the "
    "automatic retries were used up. Wait and run the agent again, run fewer "
    "agents in parallel, or use a different model. If it keeps happening, ask "
    "your administrator to check the rate limits that apply to this model."
)

_Verdict = tuple[AgentRuntimeErrorCode, UiPathErrorCategory, str, str | None]
_ResponseSignature = tuple[re.Pattern[str], _Verdict]

_BAD_REQUEST_SIGNATURES: tuple[_ResponseSignature, ...] = (
    (
        re.compile(r"too many images in request"),
        (
            AgentRuntimeErrorCode.FILE_ERROR,
            UiPathErrorCategory.USER,
            "The files sent to the model contain too many images or pages",
            "The model provider limits the number of images in one request and "
            "counts each PDF page as one image. Split large PDFs into smaller "
            "files, or analyze fewer files in one call.",
        ),
    ),
    (
        re.compile(r"image exceeds (?P<limit>\d+(?:\.\d+)?) mb"),
        (
            AgentRuntimeErrorCode.FILE_ERROR,
            UiPathErrorCategory.USER,
            "An image sent to the model is too large",
            "The model accepts images of up to {limit} MB (base64 encoding). "
            "Reduce the image's resolution or compress it, or use a different "
            "model.",
        ),
    ),
    (
        re.compile(r"maximum document size is (?P<limit>\d+(?:\.\d+)?) mb"),
        (
            AgentRuntimeErrorCode.FILE_ERROR,
            UiPathErrorCategory.USER,
            "A document sent to the model is too large",
            "The model accepts documents of up to {limit} MB. Reduce the file "
            "size or split it into smaller files, or use a different model.",
        ),
    ),
)

_NOT_FOUND_SIGNATURES: tuple[_ResponseSignature, ...] = (
    (
        re.compile(r"no active connection found for endpoint"),
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
        re.compile(r"deploymentnotfound"),
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
        re.compile(r"reached the end of its life"),
        (
            AgentRuntimeErrorCode.LLM_PROVIDER_NOT_FOUND,
            UiPathErrorCategory.DEPLOYMENT,
            "The agent's model has been retired",
            "The provider has retired the model version this agent is configured "
            "to use. Point the agent at a currently supported model.",
        ),
    ),
)

_GUARDRAIL_BLOCK_ERROR_CODE = "3040"
_GUARDRAIL_BLOCK_ERROR_CODE_HEADER = "X-UiPath-LlmGateway-ErrorCode"
_GUARDRAIL_BLOCK_DETAIL = (
    "Execution was blocked by your organization's AI Trust Layer policy."
)
_GUARDRAIL_BLOCK_MESSAGE_LIMIT = 300


def centralized_guardrail_error(error: BaseException) -> AgentRuntimeError | None:
    """The agent error for an LLM call the gateway's centralized guardrails blocked.

    The AI Trust Layer policy of the organization is enforced by the LLM gateway
    on every LLM call, and a blocked call is answered 400 with gateway error
    code 3040, in the ``X-UiPath-LlmGateway-ErrorCode`` header or in the body.
    Neither retrying nor changing the model settings helps, so this is a
    ``TERMINATION_GUARDRAIL_VIOLATION``. Its detail carries the gateway's own
    message up to the validators' details, which can quote the request.

    Args:
        error: Any exception from an LLM call: a ``UiPathAPIError``, a raw
            provider SDK error or ``httpx.HTTPStatusError`` carrying the
            response, or an exception chained from one.

    Returns:
        The error to raise, chained on ``error``, or ``None`` when ``error`` is
        not a centralized guardrail block.
    """
    api_error = as_uipath_error(error) if isinstance(error, Exception) else None
    if not isinstance(api_error, UiPathAPIError) or api_error.status_code != 400:
        return None
    if not _has_guardrail_block_error_code(api_error):
        return None

    blocked = AgentRuntimeError(
        code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION,
        title="Guardrail violation",
        detail=_guardrail_block_detail(api_error.body),
        category=UiPathErrorCategory.USER,
        status=api_error.status_code,
    )
    blocked.__cause__ = error
    return blocked


def _has_guardrail_block_error_code(error: UiPathAPIError) -> bool:
    body = error.body if isinstance(error.body, dict) else {}
    envelope = body.get("error")
    codes = (
        body.get("ErrorCode"),
        envelope.get("code") if isinstance(envelope, dict) else None,
        error.response.headers.get(_GUARDRAIL_BLOCK_ERROR_CODE_HEADER),
    )
    return any(
        isinstance(code, (int, str))
        and not isinstance(code, bool)
        and str(code).strip() == _GUARDRAIL_BLOCK_ERROR_CODE
        for code in codes
    )


def _guardrail_block_detail(body: object) -> str:
    # After the first colon come the validators' details, which can quote the request (pii_detection
    # lists the values it matched); the gateway's audit sanitiser drops them at the same place.
    for message in _gateway_messages(body):
        summary = _printable(message.partition(":")[0])
        if len(summary) > _GUARDRAIL_BLOCK_MESSAGE_LIMIT:
            summary = summary[:_GUARDRAIL_BLOCK_MESSAGE_LIMIT].rstrip() + "…"
        if summary:
            return f"{_GUARDRAIL_BLOCK_DETAIL} {summary}"
    return _GUARDRAIL_BLOCK_DETAIL


def _gateway_messages(body: object) -> list[str]:
    """The message of each gateway error envelope the body may be, verbatim."""
    if isinstance(body, str):
        return [body]
    if not isinstance(body, dict):
        return []
    error = body.get("error")
    candidates = (
        body.get("Message"),
        body.get("message"),
        error.get("message") if isinstance(error, dict) else error,
    )
    return [message for message in candidates if isinstance(message, str)]


def _printable(text: str) -> str:
    # A line break or bidi control in the gateway's message would forge lines in the error.
    return "".join(ch if ch.isprintable() else " " for ch in text).strip()


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

# Lowercased fragments of a content-safety rejection (Azure OpenAI's
# ``content_filter``), which is about policy rather than the file's format.
_CONTENT_FILTER_MARKERS = (
    "content_filter",
    "content management policy",
    "responsibleaipolicyviolation",
)


def raise_for_file_analysis_http_error(
    error: UiPathAPIError, attachments: Sequence[str], *, all_pdf: bool = False
) -> NoReturn:
    """Convert a provider rejection of a file-analysis request into an agent error.

    A 400 on a request whose only inputs are a task and the attached files is
    about those files: either the model cannot read them, or their content does
    not fit its context window. The error names the files and says which, so
    the user knows what to change -- unlike the generic bad-request detail,
    which points at model settings. A centralized guardrail block is a
    guardrail violation, not a file error. Any other status, a 400 carrying the
    gateway's own ProblemDetails ``detail`` (the gateway, not the provider,
    rejected the call), a content-filter rejection, and a 400 matching one of
    the ``_BAD_REQUEST_SIGNATURES`` (which already names the file limit hit)
    are mapped as usual by :func:`raise_for_provider_http_error`.

    The provider's own message is matched for the context-window case but never
    copied into the error: it may carry customer content, and it is already
    recorded on the LLM call span.

    Args:
        error: The normalized provider error.
        attachments: Descriptions of the files sent with the request (see
            :func:`describe_attachment`).
        all_pdf: Every attachment was sent as a PDF, so converting to PDF is
            not the advice to give.
    """
    if (blocked := centralized_guardrail_error(error)) is not None:
        raise blocked from error

    fields = _body_fields(error.body)
    if (
        error.status_code != 400
        or _gateway_detail(error.body)
        or _has_marker(fields, _CONTENT_FILTER_MARKERS)
        or _match_signature(_BAD_REQUEST_SIGNATURES, error.body) is not None
    ):
        raise_for_provider_http_error(error)

    files = ", ".join(attachments) or "the attached files"
    if _has_marker(fields, _CONTEXT_LIMIT_MARKERS):
        title = "The attached files are too large for the model"
        detail = (
            f"The content of {files} does not fit in the model's context window. "
            "Split the file or reduce its content, or use a model with a larger "
            "context window."
        )
    elif all_pdf:
        title = "The model could not read the attached files"
        detail = (
            f"The model provider rejected the request to analyze {files}. The "
            "PDF may be damaged, password-protected or have more pages than the "
            "model accepts. Check the file or use a different model, then run "
            "the agent again."
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


def _has_marker(fields: Sequence[str], markers: Sequence[str]) -> bool:
    return any(marker in field for field in fields for marker in markers)


def _gateway_detail(body: object) -> str | None:
    """The gateway's own ProblemDetails ``detail``, when it has a non-blank one."""
    detail = body.get("detail") if isinstance(body, dict) else None
    return (detail.strip() or None) if isinstance(detail, str) else None


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
    """The body's free-text fields, lowercased, for signature matching only."""
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


def _match_signature(
    signatures: tuple[_ResponseSignature, ...], body: object
) -> _Verdict | None:
    """The verdict for a body that names its own cause, else ``None``.

    The verdict's detail is a template filled with the pattern's named groups.
    """
    fields = _body_fields(body)
    for pattern, (code, category, title, detail) in signatures:
        for field in fields:
            if match := pattern.search(field):
                if detail is not None:
                    detail = detail.format_map(match.groupdict())
                return code, category, title, detail
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

    Only 400, 403 and 404 are classified beyond the 5xx/other split; 429 keeps
    that split and gains a canned detail.

    400, 403 and 404 are the statuses whose meaning depends on the body; keeping the
    code, category, title and detail decided in one place stops them drifting
    apart. They name a cause only for a body that names its own --
    ``_is_license_error`` for 403, ``_BAD_REQUEST_SIGNATURES`` for 400,
    ``_NOT_FOUND_SIGNATURES`` for 404 -- and leave the rest unnamed rather than
    guessing.
    """
    if status_code == 403:
        return _forbidden_verdict(body)

    if status_code == 400:
        if (verdict := _match_signature(_BAD_REQUEST_SIGNATURES, body)) is not None:
            return verdict
        # The relayed provider message is deliberately not quoted from the
        # body: it may carry customer PII, and it is already recorded on the
        # LLM call span, which is tenant-scoped.
        return (
            AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST,
            UiPathErrorCategory.USER,
            "LLM provider rejected the request",
            _BAD_REQUEST_DETAIL,
        )

    if (
        status_code == 404
        and (verdict := _match_signature(_NOT_FOUND_SIGNATURES, body)) is not None
    ):
        return verdict

    return (
        AgentRuntimeErrorCode.HTTP_ERROR,
        UiPathErrorCategory.SYSTEM
        if status_code >= 500
        else UiPathErrorCategory.UNKNOWN,
        f"LLM provider returned HTTP {status_code}",
        # Whether the limit is the tenant's quota or shared capacity is not in
        # the status, so the category stays as it was; only the advice is added.
        _RATE_LIMIT_DETAIL if status_code == 429 else None,
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
    return code, category, title, _gateway_detail(body) or own_detail


def raise_for_provider_http_error(error: UiPathAPIError) -> NoReturn:
    """Convert a normalized ``UiPathAPIError`` into a structured ``AgentRuntimeError``.

    Reads the HTTP status code and ``error.body``, and re-raises as an
    ``AgentRuntimeError`` chained on the original. A centralized guardrail block
    is recognised first (:func:`centralized_guardrail_error`). When
    ``_classify`` names no detail, the error's own message -- the HTTP reason
    phrase -- is all that is left.
    """
    if (blocked := centralized_guardrail_error(error)) is not None:
        raise blocked from error

    status_code = error.status_code
    code, category, title, detail = _classify(status_code, error.body)

    raise AgentRuntimeError(
        code=code,
        title=title,
        detail=detail or error.message or str(error),
        category=category,
        status=status_code,
    ) from error
