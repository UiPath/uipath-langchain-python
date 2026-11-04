"""Tests for _TraceContextHeadersCallback and per-request trace context injection."""

from unittest.mock import MagicMock, patch

from langchain_core.callbacks import BaseCallbackHandler
from uipath.core.feature_flags import FeatureFlags
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)

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
