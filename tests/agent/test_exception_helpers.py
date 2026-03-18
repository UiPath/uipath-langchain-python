"""Tests for raise_for_enriched helper."""

import json

import httpx
import pytest
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.exceptions.helpers import raise_for_enriched


def _make_enriched(
    status: int,
    body: dict | None = None,
    url: str = "https://cloud.uipath.com/org/tenant/orchestrator_/api/v1",
) -> EnrichedException:
    content = json.dumps(body).encode() if body else b""
    request = httpx.Request("POST", url)
    response = httpx.Response(
        status_code=status,
        request=request,
        headers={"content-type": "application/json"},
        content=content,
    )
    http_err = httpx.HTTPStatusError("error", request=request, response=response)
    enriched = EnrichedException(http_err)
    enriched.__cause__ = http_err
    return enriched


_KNOWN_ERRORS: dict[tuple[int, str | None], tuple[str, UiPathErrorCategory]] = {
    (404, "1002"): (
        "Could not find process for tool '{tool}'.",
        UiPathErrorCategory.USER,
    ),
    (400, "1100"): (
        "Folder not found for tool '{tool}'.",
        UiPathErrorCategory.USER,
    ),
    (409, None): (
        "Cannot start tool '{tool}': {message}",
        UiPathErrorCategory.DEPLOYMENT,
    ),
}


class TestExactMatch:
    def test_matches_status_and_error_code(self) -> None:
        err = _make_enriched(404, {"errorCode": "1002", "message": "Not found"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="MyProcess")
        assert exc_info.value.error_info.category == UiPathErrorCategory.DEPLOYMENT
        assert "MyProcess" in exc_info.value.error_info.detail

    def test_template_formatting(self) -> None:
        err = _make_enriched(404, {"errorCode": "1002", "message": "Not found"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="InvoiceBot")
        assert (
            exc_info.value.error_info.detail
            == "Could not find process for tool 'InvoiceBot'."
        )

    def test_error_code_from_http(self) -> None:
        err = _make_enriched(404, {"errorCode": "1002"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="T")
        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.HTTP_ERROR
        )


class TestWildcardMatch:
    def test_none_error_code_matches_any(self) -> None:
        err = _make_enriched(409, {"message": "Already running"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="MyTool")
        assert "Already running" in exc_info.value.error_info.detail
        assert "MyTool" in exc_info.value.error_info.detail

    def test_specific_match_takes_priority_over_wildcard(self) -> None:
        errors: dict[tuple[int, str | None], tuple[str, UiPathErrorCategory]] = {
            (404, "1002"): ("specific: {tool}", UiPathErrorCategory.DEPLOYMENT),
            (404, None): ("wildcard: {tool}", UiPathErrorCategory.SYSTEM),
        }
        err = _make_enriched(404, {"errorCode": "1002"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, errors, tool="T")
        assert exc_info.value.error_info.detail == "specific: T"
        assert exc_info.value.error_info.category == UiPathErrorCategory.DEPLOYMENT


class TestNoMatch:
    def test_unknown_status_code_does_nothing(self) -> None:
        err = _make_enriched(500, {"message": "Server error"})
        raise_for_enriched(err, _KNOWN_ERRORS, tool="T")

    def test_unknown_error_code_does_nothing(self) -> None:
        err = _make_enriched(404, {"errorCode": "9999"})
        raise_for_enriched(err, _KNOWN_ERRORS, tool="T")


class TestMessagePlaceholder:
    def test_message_from_server(self) -> None:
        err = _make_enriched(409, {"message": "Job conflict"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="T")
        assert "Job conflict" in exc_info.value.error_info.detail

    def test_empty_message_when_no_error_info(self) -> None:
        err = _make_enriched(409, body=None)
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="T")
        assert "Cannot start tool 'T': " in exc_info.value.error_info.detail


class TestMissingContext:
    def test_missing_placeholder_renders_as_unknown(self) -> None:
        err = _make_enriched(404, {"errorCode": "1002"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS)
        assert "<unknown>" in exc_info.value.error_info.detail


class TestChaining:
    def test_original_exception_is_chained(self) -> None:
        err = _make_enriched(404, {"errorCode": "1002"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="T")
        assert exc_info.value.__cause__ is err

    def test_status_code_preserved(self) -> None:
        err = _make_enriched(400, {"errorCode": "1100"})
        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_enriched(err, _KNOWN_ERRORS, tool="T")
        assert exc_info.value.error_info.status == 400
