"""Tests for _TraceContextHeadersCallback and per-request trace context injection."""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from uipath.core.feature_flags import FeatureFlags
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)

from uipath_langchain.chat._legacy.http_client import build_uipath_headers
from uipath_langchain.chat._legacy.openai import _inject_trace_context_headers
from uipath_langchain.chat.chat_model_factory import (
    _ensure_trace_context_callback,
    _TraceContextHeadersCallback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_TRACE_HEADERS = {
    "x-uipath-traceparent-id": "00-aabbccddeeff00112233445566778899-1122334455667788-01",
    "x-uipath-tracebaggage": "source=agents",
}


def _patch_build_trace_context_headers(*extra_targets: str):
    """Return a combined context-manager that patches build_trace_context_headers
    at both the canonical location and any re-import locations so tests don't
    need a live OTEL span."""
    from contextlib import ExitStack, contextmanager

    targets = [
        "uipath.platform.chat.llm_trace_context.build_trace_context_headers",
        *extra_targets,
    ]

    @contextmanager
    def _cm():
        with ExitStack() as stack:
            for t in targets:
                stack.enter_context(patch(t, return_value=dict(_MOCK_TRACE_HEADERS)))
            yield

    return _cm()


class TestTraceContextHeadersCallback:
    """The callback produces the expected headers from an active OTEL span."""

    def setup_method(self) -> None:
        FeatureFlags.configure_flags({"EnableTraceContextHeaders": True})
        set_dynamic_request_headers({})

    def teardown_method(self) -> None:
        FeatureFlags.reset_flags()
        set_dynamic_request_headers({})

    def test_returns_headers_with_active_span(self) -> None:
        mock_span = MagicMock()
        ctx = MagicMock()
        ctx.trace_id = 0xAABBCCDDEEFF00112233445566778899
        ctx.span_id = 0x1122334455667788
        mock_span.get_span_context.return_value = ctx

        cb = _TraceContextHeadersCallback()
        with (
            patch(
                "uipath.core.tracing.span_utils.UiPathSpanUtils"
                ".get_external_current_span",
                return_value=mock_span,
            ),
            patch.dict(
                "os.environ", {"UIPATH_TRACE_ID": "aabbccddeeff00112233445566778899"}
            ),
        ):
            cb._merge_headers()

        headers = get_dynamic_request_headers()
        assert "x-uipath-traceparent-id" in headers
        assert headers["x-uipath-traceparent-id"].startswith("00-")
        assert "x-uipath-tracebaggage" in headers
        assert "source=agents" in headers["x-uipath-tracebaggage"]

    def test_returns_empty_when_flag_disabled(self) -> None:
        FeatureFlags.configure_flags({"EnableTraceContextHeaders": False})
        cb = _TraceContextHeadersCallback()
        cb._merge_headers()
        assert get_dynamic_request_headers() == {}

    def test_on_chat_model_start_delegates_to_merge_headers(self) -> None:
        cb = _TraceContextHeadersCallback()
        with _patch_build_trace_context_headers(
            "uipath_langchain.chat.chat_model_factory.build_trace_context_headers",
        ):
            cb.on_chat_model_start(serialized={}, messages=[[]])
        headers = get_dynamic_request_headers()
        assert "x-uipath-traceparent-id" in headers

    def test_on_llm_start_delegates_to_merge_headers(self) -> None:
        cb = _TraceContextHeadersCallback()
        with _patch_build_trace_context_headers(
            "uipath_langchain.chat.chat_model_factory.build_trace_context_headers",
        ):
            cb.on_llm_start(serialized={}, prompts=["hello"])
        headers = get_dynamic_request_headers()
        assert "x-uipath-traceparent-id" in headers


class TestEnsureTraceContextCallback:
    """_ensure_trace_context_callback always appends the callback."""

    def test_adds_callback_to_empty_list(self) -> None:
        result = _ensure_trace_context_callback(None)
        assert any(isinstance(cb, _TraceContextHeadersCallback) for cb in result)

    def test_adds_callback_when_unset(self) -> None:
        from uipath_langchain.chat.chat_model_factory import _UNSET

        result = _ensure_trace_context_callback(_UNSET)
        assert any(isinstance(cb, _TraceContextHeadersCallback) for cb in result)

    def test_does_not_duplicate(self) -> None:
        existing: list[BaseCallbackHandler] = [_TraceContextHeadersCallback()]
        result = _ensure_trace_context_callback(existing)
        count = sum(1 for cb in result if isinstance(cb, _TraceContextHeadersCallback))
        assert count == 1

    def test_preserves_existing_callbacks(self) -> None:
        sentinel = MagicMock()
        result = _ensure_trace_context_callback([sentinel])
        assert sentinel in result
        assert any(isinstance(cb, _TraceContextHeadersCallback) for cb in result)


# ---------------------------------------------------------------------------
# Legacy OpenAI transport — per-request trace context injection
# ---------------------------------------------------------------------------


class TestOpenAITransportTraceContextHeaders:
    """_inject_trace_context_headers stamps headers on every httpx.Request."""

    def test_inject_trace_context_headers_adds_headers(self) -> None:
        request = httpx.Request("POST", "https://example.com/completions")
        with _patch_build_trace_context_headers(
            "uipath_langchain.chat._legacy.openai.build_trace_context_headers",
        ):
            _inject_trace_context_headers(request)

        for key, value in _MOCK_TRACE_HEADERS.items():
            assert request.headers[key] == value

    def test_sync_transport_calls_inject_trace_context(self) -> None:
        from uipath_langchain.chat._legacy.openai import UiPathSyncURLRewriteTransport

        transport = UiPathSyncURLRewriteTransport()
        request = httpx.Request("POST", "https://example.com/completions")

        with (
            _patch_build_trace_context_headers(
                "uipath_langchain.chat._legacy.openai.build_trace_context_headers",
            ),
            patch.object(
                httpx.HTTPTransport,
                "handle_request",
                return_value=httpx.Response(200),
            ),
        ):
            transport.handle_request(request)

        for key, value in _MOCK_TRACE_HEADERS.items():
            assert request.headers[key] == value

    async def test_async_transport_calls_inject_trace_context(self) -> None:
        from uipath_langchain.chat._legacy.openai import UiPathURLRewriteTransport

        transport = UiPathURLRewriteTransport()
        request = httpx.Request("POST", "https://example.com/completions")

        with (
            _patch_build_trace_context_headers(
                "uipath_langchain.chat._legacy.openai.build_trace_context_headers",
            ),
            patch.object(
                httpx.AsyncHTTPTransport,
                "handle_async_request",
                return_value=httpx.Response(200),
            ),
        ):
            await transport.handle_async_request(request)

        for key, value in _MOCK_TRACE_HEADERS.items():
            assert request.headers[key] == value

    def test_inject_trace_context_headers_noop_when_empty(self) -> None:
        """When build_trace_context_headers returns {}, no headers are added."""
        request = httpx.Request("POST", "https://example.com/completions")
        with patch(
            "uipath_langchain.chat._legacy.openai.build_trace_context_headers",
            return_value={},
        ):
            _inject_trace_context_headers(request)

        assert "x-uipath-traceparent-id" not in request.headers


# ---------------------------------------------------------------------------
# Legacy Vertex transport — per-request trace context injection
# ---------------------------------------------------------------------------


class TestVertexTransportTraceContextHeaders:
    """_UrlRewriteTransport and _AsyncUrlRewriteTransport inject trace headers."""

    def test_sync_transport_injects_trace_headers(self) -> None:
        pytest.importorskip("google.genai")
        from uipath_langchain.chat._legacy.vertex import _UrlRewriteTransport

        transport = _UrlRewriteTransport(
            gateway_url="https://gateway.example.com/completions"
        )
        request = httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent",
        )

        with (
            _patch_build_trace_context_headers(
                "uipath_langchain.chat._legacy.vertex.build_trace_context_headers",
            ),
            patch.object(
                httpx.HTTPTransport, "handle_request", return_value=httpx.Response(200)
            ),
        ):
            transport.handle_request(request)

        for key, value in _MOCK_TRACE_HEADERS.items():
            assert request.headers[key] == value

    async def test_async_transport_injects_trace_headers(self) -> None:
        pytest.importorskip("google.genai")
        from uipath_langchain.chat._legacy.vertex import _AsyncUrlRewriteTransport

        transport = _AsyncUrlRewriteTransport(
            gateway_url="https://gateway.example.com/completions"
        )
        request = httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent",
        )

        with (
            _patch_build_trace_context_headers(
                "uipath_langchain.chat._legacy.vertex.build_trace_context_headers",
            ),
            patch.object(
                httpx.AsyncHTTPTransport,
                "handle_async_request",
                return_value=httpx.Response(200),
            ),
        ):
            await transport.handle_async_request(request)

        for key, value in _MOCK_TRACE_HEADERS.items():
            assert request.headers[key] == value


# ---------------------------------------------------------------------------
# Legacy Bedrock transport — per-request trace context injection
# ---------------------------------------------------------------------------


class TestBedrockTransportTraceContextHeaders:
    """_modify_request in AwsBedrockCompletionsPassthroughClient injects trace headers."""

    def test_modify_request_injects_trace_headers(self) -> None:
        pytest.importorskip("botocore")
        from uipath_langchain.chat._legacy.bedrock import (
            AwsBedrockCompletionsPassthroughClient,
        )

        passthrough = AwsBedrockCompletionsPassthroughClient(
            model="anthropic.claude-haiku-4-5-20251001",
            token="test-token",
            api_flavor="converse",
        )

        class _Req:
            url = "https://bedrock.example/foo/converse"
            headers: dict[str, str] = {}

        request = _Req()
        with (
            _patch_build_trace_context_headers(
                "uipath_langchain.chat._legacy.bedrock.build_trace_context_headers",
            ),
            patch.object(
                passthrough, "_resolve_url", return_value=("https://gateway/x", False)
            ),
        ):
            passthrough._modify_request(request)

        for key, value in _MOCK_TRACE_HEADERS.items():
            assert request.headers[key] == value


# ---------------------------------------------------------------------------
# build_uipath_headers no longer includes trace context headers
# ---------------------------------------------------------------------------


class TestBuildUiPathHeadersNoTraceContext:
    """After the refactor, build_uipath_headers must NOT contain trace headers."""

    def test_no_trace_context_headers_in_build_uipath_headers(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            headers = build_uipath_headers()

        assert "x-uipath-traceparent-id" not in headers
        assert "x-uipath-tracebaggage" not in headers
