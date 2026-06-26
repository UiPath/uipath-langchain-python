"""Tests for mapping a normalized ``UiPathAPIError`` to an ``AgentRuntimeError``.

The LLM client normalizes provider HTTP errors into a ``UiPathAPIError`` carrying
``status_code`` and ``body``; ``raise_for_provider_http_error`` maps that onto the
agent's error taxonomy and surfaces the gateway ``detail``.
"""

import httpx
import pytest
from uipath.llm_client import UiPathAPIError
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.exceptions.licensing import raise_for_provider_http_error

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


def test_other_status_maps_to_http_error():
    err = _api_error(500, {"status": 500, "detail": "boom"})
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)

    info = exc_info.value.error_info
    assert info.status == 500
    assert info.category == UiPathErrorCategory.UNKNOWN
    assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
    # UNKNOWN-category errors are wrapped with a generic prefix by AgentRuntimeError,
    # but the original gateway detail is preserved within.
    assert "boom" in info.detail


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


def test_detail_falls_back_when_body_has_none():
    # Body without a "detail" key -> fall back to the error message, not crash.
    err = _api_error(403, {"status": 403})
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(err)

    info = exc_info.value.error_info
    assert info.status == 403
    assert info.detail  # non-empty (message / str fallback)
