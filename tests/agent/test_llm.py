"""Tests for mapping a normalized ``UiPathAPIError`` to an ``AgentRuntimeError``.

The LLM client normalizes provider HTTP errors into a ``UiPathAPIError`` carrying
``status_code`` and ``body``; ``raise_for_provider_http_error`` maps that onto the
agent's error taxonomy.

The load-bearing property here has cost two customer incidents: only a
positively marked gateway response is a licensing failure. PC-5000 and
SRE-654983 were both third-party 403s relayed by the gateway and reported to the
customer as LICENSE_NOT_AVAILABLE, sending them to look for AGU they already
had.

Note on ``detail``: the mapper reads the gateway's ProblemDetails ``detail`` key
-- first-party UiPath text -- and never the vendor envelope. A passthrough
provider body is therefore *not* quoted back to the customer whatever it
contained. Where the gateway supplied no ``detail``, 400 falls back to a canned,
actionable message and everything else falls back to ``UiPathAPIError.message``,
the HTTP reason phrase (an unmarked 403 reports "Forbidden"). PC-5002: the
reason-phrase fallback is what made 49% of fleet failures two words long, so the
status that dominates that bucket now carries real text that is still free of
provider content.

The tests below pin all of that down as the current contract.
"""

import httpx
import pytest
from uipath.llm_client import UiPathAPIError
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.exceptions.llm import raise_for_provider_http_error

_DETAIL = "License not available for LLM usage. You need additional 'AGU'."

# The LLM gateway's real licensing 403 (AgentHubService GlobalExceptionHandler).
_LICENSE_BODY: dict[str, object] = {
    "title": "License not available",
    "status": 403,
    "detail": _DETAIL,
    "errorCode": 10000,
    "traceId": "00-abc-def-01",
}

# A customer edge refusing the request, relayed through the BYOM passthrough
# envelope (SRE-654983).
_EDGE_HTML = (
    '<!doctype html><meta charset="utf-8">'
    '<meta name=viewport content="width=device-width, initial-scale=1">'
    "<title>403</title>403 Forbidden"
)


def _api_error(status_code: int, body: dict[str, object]) -> UiPathAPIError:
    request = httpx.Request("POST", "http://gateway/")
    response = httpx.Response(status_code, request=request, json=body)
    return UiPathAPIError.from_response(response)


def _api_error_text(status_code: int, text: str) -> UiPathAPIError:
    """A non-JSON body: from_response stores response.text verbatim."""
    request = httpx.Request("POST", "http://gateway/")
    response = httpx.Response(
        status_code,
        request=request,
        text=text,
        headers={"content-type": "text/html"},
    )
    return UiPathAPIError.from_response(response)


def _raise(err: UiPathAPIError) -> AgentRuntimeError:
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)
    return exc_info.value


# --------------------------------------------------------------------------
# Licensing: only a positively marked gateway response
# --------------------------------------------------------------------------


def test_403_with_gateway_marker_maps_to_license_not_available():
    info = _raise(_api_error(403, _LICENSE_BODY)).error_info

    assert info.status == 403
    assert info.category == UiPathErrorCategory.DEPLOYMENT
    assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)
    assert info.title == "License not available"
    assert info.detail == _DETAIL


def test_403_with_error_code_only_maps_to_license_not_available():
    err = _api_error(403, {"status": 403, "detail": _DETAIL, "errorCode": 10000})
    info = _raise(err).error_info

    assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)
    # No title in the body -> the mapper supplies its own.
    assert info.title == "License not available"


def test_403_with_title_only_maps_to_license_not_available():
    # Automation Suite builds predating errorCode, and the legacy client path,
    # carry the title marker alone.
    err = _api_error(403, {"title": "License not available", "detail": _DETAIL})
    info = _raise(err).error_info

    assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)


@pytest.mark.parametrize(
    "error_code",
    [pytest.param(10000, id="int"), pytest.param("10000", id="numeric-string")],
)
def test_license_error_code_is_read_as_int_or_string(error_code):
    err = _api_error(403, {"detail": _DETAIL, "errorCode": error_code})

    info = _raise(err).error_info
    assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)


def test_non_numeric_error_code_does_not_crash_the_mapper():
    # A provider that puts a symbolic code where an int was expected must not
    # turn a 403 into an unhandled ValueError inside the error mapper itself.
    err = _api_error(403, {"detail": "no", "errorCode": "PERMISSION_DENIED"})

    info = _raise(err).error_info
    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_FORBIDDEN.value)


def test_license_detail_is_kept_in_telemetry():
    # First-party UiPath prose: withholding it would blind real licensing triage.
    error = _raise(_api_error(403, _LICENSE_BODY))

    assert _DETAIL in str(error)


# --------------------------------------------------------------------------
# Non-licensing 403s
# --------------------------------------------------------------------------


def test_403_with_other_error_code_is_not_licensing():
    # 10900 is AgentHubErrorCode.Forbidden -- an authorization denial. It is a
    # marked gateway response, but not a *licensing* one.
    err = _api_error(403, {"title": "Forbidden", "detail": "no", "errorCode": 10900})
    info = _raise(err).error_info

    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_FORBIDDEN.value)
    assert info.title == "LLM provider returned HTTP 403"
    # A gateway-supplied detail is still surfaced.
    assert info.detail == "no"


@pytest.mark.parametrize(
    "err_factory",
    [
        pytest.param(
            lambda: _api_error(403, {"error": {"message": _EDGE_HTML}}),
            id="byom-envelope",
        ),
        pytest.param(lambda: _api_error_text(403, _EDGE_HTML), id="raw-html"),
        pytest.param(lambda: _api_error_text(403, ""), id="empty-body"),
        pytest.param(
            lambda: _api_error(403, {"weird": "shape", "code": 7}),
            id="unrecognised-dict",
        ),
        pytest.param(
            lambda: _api_error(403, {"status": 403}),
            id="problem-details-without-detail",
        ),
    ],
)
def test_unmarked_403_maps_to_provider_forbidden(err_factory):
    # Regression for PC-5000 / SRE-654983. None of these bodies carries the
    # gateway's licensing marker, so none of them may be reported as licensing --
    # whether the body is a passthrough envelope, a raw HTML edge page, empty, or
    # simply an unrecognised shape.
    info = _raise(err_factory()).error_info

    assert info.status == 403
    assert info.category == UiPathErrorCategory.DEPLOYMENT
    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_FORBIDDEN.value)
    # No ProblemDetails detail to read -> the HTTP reason phrase, not the body.
    assert info.detail == "Forbidden"


@pytest.mark.parametrize(
    "err_factory",
    [
        pytest.param(
            lambda: _api_error(403, {"error": {"message": _EDGE_HTML}}),
            id="byom-envelope",
        ),
        pytest.param(lambda: _api_error_text(403, _EDGE_HTML), id="raw-html"),
    ],
)
def test_provider_body_is_not_copied_into_the_agent_error(err_factory):
    # The upstream body is third-party content of unknown sensitivity. It reaches
    # neither the customer-facing contract nor str(exc) -- the latter is what
    # span.record_exception() and AgentRun.Failed read.
    #
    # Scoped to this exception, not the chain: the chained UiPathAPIError still
    # renders its own raw body via its __str__.
    error = _raise(err_factory())

    for rendered in (error.error_info.detail, str(error), repr(error)):
        assert "403 Forbidden" not in rendered
        assert "doctype" not in rendered.lower()


# --------------------------------------------------------------------------
# Other statuses
# --------------------------------------------------------------------------


def test_5xx_maps_to_system_http_error():
    info = _raise(_api_error(500, {"status": 500, "detail": "boom"})).error_info

    assert info.status == 500
    assert info.category == UiPathErrorCategory.SYSTEM
    assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
    assert info.title == "LLM provider returned HTTP 500"
    assert "boom" in info.detail


@pytest.mark.parametrize("status_code", [404, 408, 413, 422, 429])
def test_unclassified_4xx_remains_unknown(status_code: int):
    """Only 400 and 403 are classified; the rest of 4xx stays UNKNOWN.

    404 is here on purpose. Every LLM-gateway 404 in prd over 30 days was a
    missing or unreachable deployment -- BYO relay not connected, Azure
    ``DeploymentNotFound``, a retired Bedrock model -- i.e. Deployment, not
    User. It is left UNKNOWN until that is decided on its own evidence rather
    than folded into the 400 change.
    """
    err = _api_error(status_code, {"status": status_code, "detail": "nope"})
    info = _raise(err).error_info

    assert info.category == UiPathErrorCategory.UNKNOWN
    assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
    assert info.title == f"LLM provider returned HTTP {status_code}"


# --------------------------------------------------------------------------
# 400: User, with a canned detail instead of the reason phrase
# --------------------------------------------------------------------------

# The body of the 400 that failed 192/192 runs on gpt-4.1-mini-e2e-custom
# (job 1fab7e97-...): max_tokens=65535 written by Agent Builder itself.
_MAX_TOKENS_BODY: dict[str, object] = {
    "error": {
        "message": (
            "max_tokens is too large: 65535. This model supports at most 32768 "
            "completion tokens, whereas you provided 65535."
        ),
        "code": "invalid_value",
        "param": "max_tokens",
    }
}


@pytest.mark.parametrize(
    "err_factory",
    [
        pytest.param(lambda: _api_error(400, _MAX_TOKENS_BODY), id="vendor-envelope"),
        pytest.param(
            lambda: _api_error(400, {"message": "Malformed input request."}),
            id="bedrock-envelope",
        ),
        pytest.param(lambda: _api_error_text(400, _EDGE_HTML), id="raw-html"),
        pytest.param(lambda: _api_error(400, {}), id="empty-body"),
    ],
)
def test_400_maps_to_user_with_a_canned_detail(err_factory):
    info = _raise(err_factory()).error_info

    assert info.status == 400
    assert info.category == UiPathErrorCategory.USER
    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST.value)
    assert info.title == "LLM provider rejected the request"
    # The bare reason phrase is what PC-5002 is about -- it must be gone.
    assert info.detail != "Bad Request"
    assert "model settings" in info.detail


@pytest.mark.parametrize(
    "err_factory",
    [
        pytest.param(lambda: _api_error(400, _MAX_TOKENS_BODY), id="vendor-envelope"),
        pytest.param(lambda: _api_error_text(400, _EDGE_HTML), id="raw-html"),
    ],
)
def test_400_does_not_quote_the_provider_body(err_factory):
    error = _raise(err_factory())

    for rendered in (error.error_info.detail, str(error), repr(error)):
        assert "65535" not in rendered
        assert "doctype" not in rendered.lower()


def test_400_prefers_the_gateway_detail_over_the_canned_text():
    """A ProblemDetails ``detail`` is first-party UiPath text and more specific."""
    err = _api_error(400, {"status": 400, "detail": "Model not enabled."})
    info = _raise(err).error_info

    assert info.detail == "Model not enabled."
    assert info.category == UiPathErrorCategory.USER
    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST.value)


def test_user_category_is_not_wrapped_in_the_generic_prefix():
    """USER is outside _SHOULD_WRAP_CATEGORIES, so the canned detail stands alone."""
    info = _raise(_api_error(400, _MAX_TOKENS_BODY)).error_info

    assert not info.detail.startswith("An unexpected error occurred")
    assert info.detail.startswith("The model provider rejected the request")


def test_legacy_raw_provider_error_is_normalized_and_mapped():
    # Legacy clients (use_new_llm_clients=False) raise raw provider SDK exceptions,
    # not UiPathAPIError. as_uipath_error normalizes them so licensing still maps.
    import openai
    from uipath.llm_client.utils.exceptions import as_uipath_error

    request = httpx.Request("POST", "http://gateway/")
    response = httpx.Response(403, request=request, json=_LICENSE_BODY)
    raw = openai.PermissionDeniedError(
        "Forbidden", response=response, body=_LICENSE_BODY
    )

    uipath_error = as_uipath_error(raw)
    assert isinstance(uipath_error, UiPathAPIError)
    info = _raise(uipath_error).error_info

    assert info.status == 403
    assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)
    assert info.detail == _DETAIL
