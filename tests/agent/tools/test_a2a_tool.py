"""Tests for the A2A tool: URL resolution and trace-context propagation."""

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import AgentCard
from uipath.agent.models.agent import AgentA2aResourceConfig

from uipath_langchain.agent.tools.a2a.a2a_tool import (
    A2aClient,
    _build_client_headers,
    _build_traceparent,
    _coerce_span_id,
    _normalize_trace_id,
    _resolve_a2a_url,
    create_a2a_tools_and_clients,
)

PROXY_URL = "https://cloud.uipath.com/proxy/a2a/remote-agent"
CACHED_URL = "https://internal.example.com/agents/remote-agent"


def _make_resource(
    *,
    a2a_url: str = PROXY_URL,
    cached_agent_card: dict[str, Any] | None = None,
    is_enabled: bool = True,
) -> AgentA2aResourceConfig:
    """Build an A2A resource config for tests."""
    return AgentA2aResourceConfig(
        id="resource-id",
        name="remote-agent",
        description="A remote A2A agent",
        is_enabled=is_enabled,
        slug="remote-agent-slug",
        folder_path="Shared",
        a2a_url=a2a_url,
        cached_agent_card=cached_agent_card,
    )


def test_resolve_a2a_url_returns_proxy_url() -> None:
    resource = _make_resource(a2a_url=PROXY_URL)
    assert _resolve_a2a_url(resource) == PROXY_URL


def test_resolve_a2a_url_raises_when_missing() -> None:
    resource = _make_resource(a2a_url="")
    with pytest.raises(ValueError, match="has no URL"):
        _resolve_a2a_url(resource)


def test_create_tools_prefers_proxy_url_over_cached_card_url() -> None:
    """The proxy a2a_url must override the URL embedded in the cached card."""
    resource = _make_resource(
        a2a_url=PROXY_URL,
        cached_agent_card={
            "url": CACHED_URL,
            "name": "Remote Agent",
            "description": "cached",
            "version": "1.0.0",
            "skills": [],
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    )

    tools, clients = create_a2a_tools_and_clients([resource])

    assert len(tools) == 1
    assert len(clients) == 1
    # The client's agent card URL is rewritten to the proxy URL.
    assert clients[0]._agent_card.url == PROXY_URL


def test_create_tools_sets_url_when_no_cached_card() -> None:
    resource = _make_resource(a2a_url=PROXY_URL, cached_agent_card=None)

    tools, clients = create_a2a_tools_and_clients([resource])

    assert len(tools) == 1
    assert clients[0]._agent_card.url == PROXY_URL


def test_create_tools_skips_disabled_resource() -> None:
    resource = _make_resource(is_enabled=False)

    tools, clients = create_a2a_tools_and_clients([resource])

    assert tools == []
    assert clients == []


_GUID = "12345678-9abc-def0-1234-56789abcdef0"
_HEX32 = "123456789abcdef0123456789abcdef0"
_SPAN16 = "fedcba9876543210"
_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-01$")


class TestNormalizeTraceId:
    def test_strips_dashes_and_lowercases_uuid(self):
        assert _normalize_trace_id(_GUID.upper()) == _HEX32

    def test_passes_through_hex(self):
        assert _normalize_trace_id(_HEX32) == _HEX32

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            _normalize_trace_id("deadbeef")


class TestCoerceSpanId:
    def test_accepts_16_hex(self):
        assert _coerce_span_id(_SPAN16) == _SPAN16

    def test_lowercases(self):
        assert _coerce_span_id(_SPAN16.upper()) == _SPAN16

    def test_rejects_none_and_empty(self):
        assert _coerce_span_id(None) is None
        assert _coerce_span_id("") is None

    def test_rejects_wrong_length(self):
        assert _coerce_span_id("abc") is None

    def test_rejects_non_hex(self):
        assert _coerce_span_id("zzzzzzzzzzzzzzzz") is None


class TestBuildTraceparent:
    def test_none_without_trace_id(self, monkeypatch):
        monkeypatch.delenv("UIPATH_TRACE_ID", raising=False)
        assert _build_traceparent() is None

    def test_none_with_invalid_trace_id(self, monkeypatch):
        monkeypatch.setenv("UIPATH_TRACE_ID", "not-a-trace-id")
        assert _build_traceparent() is None

    def test_uses_trace_id_and_parent_span(self, monkeypatch):
        monkeypatch.setenv("UIPATH_TRACE_ID", _HEX32)
        monkeypatch.setenv("UIPATH_PARENT_SPAN_ID", _SPAN16)
        assert _build_traceparent() == f"00-{_HEX32}-{_SPAN16}-01"

    def test_normalizes_uuid_trace_id(self, monkeypatch):
        monkeypatch.setenv("UIPATH_TRACE_ID", _GUID)
        monkeypatch.setenv("UIPATH_PARENT_SPAN_ID", _SPAN16)
        assert _build_traceparent() == f"00-{_HEX32}-{_SPAN16}-01"

    def test_mints_parent_when_absent(self, monkeypatch):
        monkeypatch.setenv("UIPATH_TRACE_ID", _HEX32)
        monkeypatch.delenv("UIPATH_PARENT_SPAN_ID", raising=False)
        result = _build_traceparent()
        assert result is not None
        assert _TRACEPARENT_RE.match(result)
        assert result.startswith(f"00-{_HEX32}-")


class TestBuildClientHeaders:
    def test_authorization_only_without_job_context(self, monkeypatch):
        monkeypatch.delenv("UIPATH_TRACE_ID", raising=False)
        monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
        headers = _build_client_headers("tok")
        assert headers["Authorization"] == "Bearer tok"
        assert "traceparent" not in headers
        assert "X-UiPath-JobKey" not in headers

    def test_includes_traceparent_and_job_key(self, monkeypatch):
        monkeypatch.setenv("UIPATH_TRACE_ID", _HEX32)
        monkeypatch.setenv("UIPATH_PARENT_SPAN_ID", _SPAN16)
        monkeypatch.setenv("UIPATH_JOB_KEY", "job-123")
        headers = _build_client_headers("tok")
        assert headers["Authorization"] == "Bearer tok"
        assert headers["traceparent"] == f"00-{_HEX32}-{_SPAN16}-01"
        assert headers["X-UiPath-JobKey"] == "job-123"


class TestA2aClientGet:
    async def test_get_applies_trace_and_job_headers(self, monkeypatch):
        monkeypatch.setenv("UIPATH_TRACE_ID", _HEX32)
        monkeypatch.setenv("UIPATH_PARENT_SPAN_ID", _SPAN16)
        monkeypatch.setenv("UIPATH_JOB_KEY", "job-123")

        card = AgentCard(
            url="https://example.test/a2a",
            name="agent",
            description="",
            version="1.0.0",
            skills=[],
            capabilities={},
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
        )
        client = A2aClient(card)
        sdk = MagicMock()
        sdk._config.secret = "tok"

        with (
            patch("uipath.platform.UiPath", return_value=sdk),
            patch(
                "a2a.client.ClientFactory.connect",
                new=AsyncMock(return_value=MagicMock()),
            ),
        ):
            await client.get()

        try:
            assert client._http_client is not None
            headers = client._http_client.headers
            assert headers["authorization"] == "Bearer tok"
            assert headers["traceparent"] == f"00-{_HEX32}-{_SPAN16}-01"
            assert headers["x-uipath-jobkey"] == "job-123"
        finally:
            await client.dispose()
