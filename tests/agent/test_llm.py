"""Tests for mapping a normalized ``UiPathAPIError`` to an ``AgentRuntimeError``.

The LLM client normalizes provider HTTP errors into a ``UiPathAPIError`` carrying
``status_code`` and ``body``; ``raise_for_provider_http_error`` maps that onto the
agent's error taxonomy.

The load-bearing property here has cost two customer incidents: only a
positively marked gateway response is a licensing failure. PC-5000 and
SRE-654983 were both third-party 403s relayed by the gateway and reported to the
customer as LICENSE_NOT_AVAILABLE, sending them to look for AGU they already
had.

Note on ``detail``: the mapper quotes only the gateway's ProblemDetails
``detail`` key -- first-party UiPath text -- and never the vendor envelope. A
passthrough provider body is therefore *not* quoted back to the customer
whatever it contained. 404 reads the envelope to *classify* -- a body that names
a missing model, deployment or relay is Deployment, and anything else keeps the
5xx/other split -- but still emits its own text.
Where the gateway supplied no ``detail``, 400 and a named 404 fall back to a canned,
actionable message and everything else falls back to ``UiPathAPIError.message``,
the HTTP reason phrase (an unmarked 403 reports "Forbidden"). The reason-phrase
fallback is what made 49% of fleet failures two words long, so the statuses that
dominate that bucket now carry real text that is still free of provider content.

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


@pytest.mark.parametrize("status_code", [408, 413, 422, 429])
def test_unclassified_4xx_remains_unknown(status_code: int):
    """Only 400, 403 and 404 are classified; the rest of 4xx stays UNKNOWN."""
    err = _api_error(status_code, {"status": status_code, "detail": "nope"})
    info = _raise(err).error_info

    assert info.category == UiPathErrorCategory.UNKNOWN
    assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
    assert info.title == f"LLM provider returned HTTP {status_code}"


# --------------------------------------------------------------------------
# 400: User, with a canned detail instead of the reason phrase
# --------------------------------------------------------------------------

# The body of the 400 that failed 192/192 runs on gpt-4.1-mini-e2e-custom:
# max_tokens=65535 written by Agent Builder itself.
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
    # The bare reason phrase is the failure mode being fixed -- it must be gone.
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


# --------------------------------------------------------------------------
# 404: named only where the body names itself, and never User
# --------------------------------------------------------------------------
#
# The bodies below are the whole agent-attributable 404 population of prd over
# 30 days (24 events on Agents.* / AgentHub.LLM / ConversationalAgents.*
# operation codes). Three signatures cover 17 of them; the remaining 7 are left
# UNKNOWN rather than given a cause the response never claimed -- the mistake
# PC-5000 and SRE-654983 were about.

# Integration Service, when a bring-your-own-model connection's relay client is
# not connected. 7 events, on gpt-4.1-AMD-LLMGateway / gpt-5.4-AMD-LLMGateway.
# It flaps: the same connection answered OK 21 seconds before returning this.
_RELAY_DOWN_BODY: dict[str, object] = {
    "error": {
        "message": (
            "No active connection found for endpoint. Ensure the relay client is "
            "running and connected for this endpoint. If you recently updated the "
            "relay configuration, perform relay reload on your nodes to pick up "
            "the changes."
        )
    }
}

# Azure OpenAI, when the deployment behind a BYO model is gone. 6 events, the
# largest single signature. The marker is in ``error.code``, not the message.
_AZURE_DEPLOYMENT_BODY: dict[str, object] = {
    "error": {
        "type": "invalid_request_error",
        "code": "DeploymentNotFound",
        "message": (
            "The API deployment for this resource does not exist. If you created "
            "the deployment within the last 5 minutes, please wait a moment and "
            "try again."
        ),
    }
}

# Bedrock, when the configured model version has been retired. 4 events. The
# marker is a top-level ``message`` with no ``error`` envelope at all.
_BEDROCK_RETIRED_BODY: dict[str, object] = {
    "message": (
        "This model version has reached the end of its life. Please refer to the "
        "AWS documentation for more details."
    )
}

# The gateway logs "No llm configuration found" and forwards anyway, to a
# publisher that does not host the model -- gemini-3.7-flash to
# publishers/anthropic, claude-opus-5 to publishers/google. 1 event in this
# window, non-BYO. Deliberately *not* a signature: it names a publisher, not a
# model deployment, and the marker would be a Vertex-specific phrase we would be
# guessing at from a single event.
_VERTEX_PUBLISHER_BODY: dict[str, object] = {
    "error": {
        "code": 404,
        "message": (
            "Publisher model `projects/uipath-llm-gateway-prd/locations/us/"
            "publishers/anthropic/models/gemini-3.7-flash` was not found or your "
            "project does not have access to it."
        ),
        "status": "NOT_FOUND",
    }
}

# The 6 that name nothing: an empty body (3), "The operation was canceled." (2),
# and a bare "Resource not found" (1).
_UNNAMEABLE_404_BODIES = [
    pytest.param(lambda: _api_error_text(404, ""), id="empty-body"),
    pytest.param(lambda: _api_error_text(404, " "), id="whitespace-body"),
    pytest.param(
        lambda: _api_error_text(404, "The operation was canceled."), id="canceled"
    ),
    pytest.param(
        lambda: _api_error(
            404, {"error": {"code": "404", "message": "Resource not found"}}
        ),
        id="bare-resource-not-found",
    ),
    pytest.param(lambda: _api_error_text(404, _EDGE_HTML), id="raw-html"),
]

# Everything the mapper leaves UNKNOWN: the bodies that name nothing, plus the
# publisher mismatch, which names something we deliberately do not key on.
_UNCLASSIFIED_404_BODIES = [
    *_UNNAMEABLE_404_BODIES,
    pytest.param(
        lambda: _api_error(404, _VERTEX_PUBLISHER_BODY), id="vertex-publisher"
    ),
]

_ALL_404_BODIES = [
    pytest.param(lambda: _api_error(404, _RELAY_DOWN_BODY), id="relay-down"),
    pytest.param(
        lambda: _api_error(404, _AZURE_DEPLOYMENT_BODY), id="azure-deployment"
    ),
    pytest.param(lambda: _api_error(404, _BEDROCK_RETIRED_BODY), id="bedrock-retired"),
    *_UNCLASSIFIED_404_BODIES,
]


def test_404_with_relay_marker_maps_to_deployment():
    info = _raise(_api_error(404, _RELAY_DOWN_BODY)).error_info

    assert info.status == 404
    assert info.category == UiPathErrorCategory.DEPLOYMENT
    assert info.code.endswith(
        AgentRuntimeErrorCode.LLM_BYO_CONNECTION_UNAVAILABLE.value
    )
    assert "relay" in info.detail.lower()


def test_404_with_azure_deployment_marker_maps_to_deployment():
    """The marker is ``error.code``; the message never says "not found"."""
    info = _raise(_api_error(404, _AZURE_DEPLOYMENT_BODY)).error_info

    assert info.category == UiPathErrorCategory.DEPLOYMENT
    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_NOT_FOUND.value)
    assert "deployment" in info.title.lower()
    assert "bring your own" in info.detail.lower()


def test_404_with_retired_model_marker_maps_to_deployment():
    """A top-level ``message`` with no ``error`` envelope -- Bedrock's shape."""
    info = _raise(_api_error(404, _BEDROCK_RETIRED_BODY)).error_info

    assert info.category == UiPathErrorCategory.DEPLOYMENT
    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_NOT_FOUND.value)
    assert "retired" in info.detail.lower()


def test_404_marker_is_matched_in_a_non_json_body():
    err = _api_error_text(
        404, "No active connection found for endpoint. Ensure the relay client"
    )
    info = _raise(err).error_info

    assert info.category == UiPathErrorCategory.DEPLOYMENT


@pytest.mark.parametrize("err_factory", _UNCLASSIFIED_404_BODIES)
def test_404_without_a_recognized_marker_stays_unknown(err_factory):
    """No marker we classify, no cause. UNKNOWN is honest, a guess is not."""
    info = _raise(err_factory()).error_info

    assert info.status == 404
    assert info.category == UiPathErrorCategory.UNKNOWN
    assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
    assert info.title == "LLM provider returned HTTP 404"


@pytest.mark.parametrize("err_factory", _ALL_404_BODIES)
def test_404_is_never_a_user_error(err_factory):
    """Not one agent-attributable LLM 404 in prd was caused by the user.

    A missing BYO deployment, a retired model and a disconnected relay are all
    things an administrator fixes; the bodies left UNKNOWN must not be pinned on
    the user either.
    """
    info = _raise(err_factory()).error_info

    assert info.category != UiPathErrorCategory.USER


@pytest.mark.parametrize("err_factory", _ALL_404_BODIES)
def test_404_does_not_quote_the_provider_body(err_factory):
    """The body is read to classify, never to quote.

    A BYO passthrough relays a customer-controlled endpoint, so this position is
    third-party content of unknown sensitivity whatever it happens to say.
    """
    error = _raise(err_factory())

    for rendered in (error.error_info.detail, str(error), repr(error)):
        assert "uipath-llm-gateway-prd" not in rendered
        assert "within the last 5 minutes" not in rendered
        assert "AWS documentation" not in rendered
        assert "perform relay reload on your nodes" not in rendered
        assert "doctype" not in rendered.lower()


def test_404_prefers_the_gateway_detail_over_the_canned_text():
    """A ProblemDetails ``detail`` is first-party UiPath text and more specific."""
    err = _api_error(404, {"status": 404, "detail": "Model not enabled for tenant."})
    info = _raise(err).error_info

    assert "Model not enabled for tenant." in info.detail
