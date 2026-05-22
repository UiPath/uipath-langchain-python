"""Tests for _TraceContextHeadersCallback in chat_model_factory."""

from unittest.mock import MagicMock, patch

from langchain_core.callbacks import BaseCallbackHandler
from uipath.core.feature_flags import FeatureFlags
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)

from uipath_langchain.chat.chat_model_factory import (
    _TraceContextHeadersCallback,
    _ensure_trace_context_callback,
)


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
        with patch(
            "uipath.core.tracing.span_utils.UiPathSpanUtils"
            ".get_external_current_span",
            return_value=mock_span,
        ), patch.dict(
            "os.environ", {"UIPATH_TRACE_ID": "aabbccddeeff00112233445566778899"}
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
