"""Tests for mapping a normalized ``UiPathAPIError`` to an ``AgentRuntimeError``.

The LLM client normalizes provider HTTP errors into a ``UiPathAPIError`` carrying
``status_code`` and ``body``; ``raise_for_provider_http_error`` maps that onto the
agent's error taxonomy and surfaces the gateway ``detail``.
"""

import traceback

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


def _api_error(status_code: int, body: dict[str, object]) -> UiPathAPIError:
    request = httpx.Request("POST", "http://gateway/")
    response = httpx.Response(status_code, request=request, json=body)
    return UiPathAPIError.from_response(response)


def test_403_maps_to_license_not_available():
    err = _api_error(403, {"status": 403, "detail": _DETAIL})
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)

    info = exc_info.value.error_info
    assert info.status == 403
    assert info.category == UiPathErrorCategory.DEPLOYMENT
    assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)
    assert info.detail == _DETAIL


def test_5xx_maps_to_system_http_error():
    err = _api_error(500, {"status": 500, "detail": "boom"})
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)

    info = exc_info.value.error_info
    assert info.status == 500
    assert info.category == UiPathErrorCategory.SYSTEM
    assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
    # SYSTEM-category errors are wrapped with a generic prefix by AgentRuntimeError,
    # but the original gateway detail is preserved within.
    assert "boom" in info.detail


def test_unclassified_status_remains_unknown():
    err = _api_error(400, {"status": 400, "detail": "bad request"})
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)

    assert exc_info.value.error_info.category == UiPathErrorCategory.UNKNOWN


def test_legacy_raw_provider_error_is_normalized_and_mapped():
    # Legacy clients (use_new_llm_clients=False) raise raw provider SDK exceptions,
    # not UiPathAPIError. as_uipath_error normalizes them so licensing still maps.
    import openai
    from uipath.llm_client.utils.exceptions import as_uipath_error

    request = httpx.Request("POST", "http://gateway/")
    response = httpx.Response(
        403, request=request, json={"status": 403, "detail": _DETAIL}
    )
    raw = openai.PermissionDeniedError(
        "Forbidden", response=response, body={"status": 403, "detail": _DETAIL}
    )

    uipath_error = as_uipath_error(raw)
    assert isinstance(uipath_error, UiPathAPIError)
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(uipath_error)

    info = exc_info.value.error_info
    assert info.status == 403
    assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)
    assert info.detail == _DETAIL


def test_vendor_message_never_in_detail():
    # A vendor passthrough message can echo request content, so it never lands in the
    # run detail (which reaches App Insights) — it stays on the trace span only.
    err = _api_error(400, {"message": "`temperature` may only be set to 1"})
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)

    info = exc_info.value.error_info
    assert info.status == 400
    assert "temperature" not in info.detail
    assert info.detail  # non-empty generic message


def test_gateway_detail_always_shown():
    # The gateway's own {"detail": ...} (e.g. licensing) is UiPath text, always shown.
    err = _api_error(403, {"detail": _DETAIL})
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)

    assert exc_info.value.error_info.detail == _DETAIL


def test_provider_body_absent_from_traceback():
    # format_exc() is logged to App Insights as ErrorTraceback; the vendor body must
    # not ride along via the chained UiPathAPIError string. from None suppresses it.
    err = _api_error(400, {"message": "secret PII in the model error"})
    try:
        try:
            raise err
        except UiPathAPIError as caught:
            raise_for_provider_http_error(caught)
    except AgentRuntimeError:
        tb = traceback.format_exc()
    assert "secret PII in the model error" not in tb
