"""Tests for BedrockInvokePayloadHandler and BedrockConversePayloadHandler."""

import logging

import pytest
from langchain_core.messages import AIMessage

from uipath_langchain.chat.exceptions import ChatModelError
from uipath_langchain.chat.handlers.bedrock import (
    BedrockConversePayloadHandler,
    BedrockInvokePayloadHandler,
)

# ---------------------------------------------------------------------------
# Fake model factories
# ---------------------------------------------------------------------------


def make_invoke_model(**model_kwargs_override: object) -> object:
    """Return a ChatBedrock-like model with optional model_kwargs."""
    model = type("FakeChatBedrock", (), {"model_kwargs": {}})()
    model.model_kwargs = model_kwargs_override
    return model


def make_converse_model(**fields_override: object) -> object:
    """Return a ChatBedrockConverse-like model with optional additional_model_request_fields."""
    model = type(
        "FakeChatBedrockConverse", (), {"additional_model_request_fields": {}}
    )()
    model.additional_model_request_fields = fields_override
    return model


def make_thinking_invoke_model() -> object:
    return make_invoke_model(thinking={"type": "enabled"})


def make_thinking_converse_model() -> object:
    return make_converse_model(thinking={"type": "enabled"})


# ---------------------------------------------------------------------------
# BedrockInvokePayloadHandler — get_tool_binding_kwargs
# ---------------------------------------------------------------------------


class TestBedrockInvokeGetToolBindingKwargs:
    def setup_method(self) -> None:
        self.handler = BedrockInvokePayloadHandler(make_invoke_model())  # type: ignore[arg-type]

    def test_auto_no_flags(self) -> None:
        result = self.handler.get_tool_binding_kwargs([], "auto")
        assert result == {"tool_choice": "auto"}

    def test_any_parallel_true(self) -> None:
        result = self.handler.get_tool_binding_kwargs(
            [], "any", parallel_tool_calls=True
        )
        assert result == {"tool_choice": "any"}

    def test_any_parallel_false_sets_disable_flag(self) -> None:
        result = self.handler.get_tool_binding_kwargs(
            [], "any", parallel_tool_calls=False
        )
        assert result == {"tool_choice": "any", "disable_parallel_tool_use": True}

    def test_auto_parallel_false_sets_disable_flag(self) -> None:
        result = self.handler.get_tool_binding_kwargs(
            [], "auto", parallel_tool_calls=False
        )
        assert result == {"tool_choice": "auto", "disable_parallel_tool_use": True}

    def test_strict_mode_ignored(self) -> None:
        result = self.handler.get_tool_binding_kwargs([], "auto", strict_mode=True)
        assert "strict" not in result

    def test_thinking_downgrades_any_to_auto(self) -> None:
        handler = BedrockInvokePayloadHandler(make_thinking_invoke_model())  # type: ignore[arg-type]
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "auto"

    def test_thinking_does_not_downgrade_auto(self) -> None:
        handler = BedrockInvokePayloadHandler(make_thinking_invoke_model())  # type: ignore[arg-type]
        result = handler.get_tool_binding_kwargs([], "auto")
        assert result["tool_choice"] == "auto"

    def test_thinking_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = BedrockInvokePayloadHandler(make_thinking_invoke_model())  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING):
            handler.get_tool_binding_kwargs([], "any")
        assert any("tool_choice" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# BedrockConversePayloadHandler — get_tool_binding_kwargs
# ---------------------------------------------------------------------------


class TestBedrockConverseGetToolBindingKwargs:
    def setup_method(self) -> None:
        self.handler = BedrockConversePayloadHandler(make_converse_model())  # type: ignore[arg-type]

    def test_auto_no_flags(self) -> None:
        result = self.handler.get_tool_binding_kwargs([], "auto")
        assert result == {"tool_choice": "auto"}

    def test_any_no_flags(self) -> None:
        result = self.handler.get_tool_binding_kwargs([], "any")
        assert result == {"tool_choice": "any"}

    def test_strict_mode_included(self) -> None:
        result = self.handler.get_tool_binding_kwargs([], "auto", strict_mode=True)
        assert result == {"tool_choice": "auto", "strict": True}

    def test_any_strict_mode_included(self) -> None:
        result = self.handler.get_tool_binding_kwargs([], "any", strict_mode=True)
        assert result == {"tool_choice": "any", "strict": True}

    def test_parallel_tool_calls_ignored(self) -> None:
        result = self.handler.get_tool_binding_kwargs(
            [], "auto", parallel_tool_calls=False
        )
        assert "disable_parallel_tool_use" not in result

    def test_thinking_downgrades_any_to_auto(self) -> None:
        handler = BedrockConversePayloadHandler(make_thinking_converse_model())  # type: ignore[arg-type]
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "auto"

    def test_thinking_does_not_downgrade_auto(self) -> None:
        handler = BedrockConversePayloadHandler(make_thinking_converse_model())  # type: ignore[arg-type]
        result = handler.get_tool_binding_kwargs([], "auto")
        assert result["tool_choice"] == "auto"

    def test_thinking_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = BedrockConversePayloadHandler(make_thinking_converse_model())  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING):
            handler.get_tool_binding_kwargs([], "any")
        assert any("tool_choice" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# BedrockInvokePayloadHandler — check_stop_reason
# ---------------------------------------------------------------------------


class TestBedrockInvokeCheckStopReason:
    def setup_method(self) -> None:
        self.handler = BedrockInvokePayloadHandler(make_invoke_model())  # type: ignore[arg-type]

    def test_stop_sequence_no_raise(self) -> None:
        msg = AIMessage(
            content="ok", response_metadata={"stop_reason": "stop_sequence"}
        )
        self.handler.check_stop_reason(msg)

    def test_max_tokens_raises(self) -> None:
        msg = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
        with pytest.raises(ChatModelError):
            self.handler.check_stop_reason(msg)

    def test_refusal_raises(self) -> None:
        msg = AIMessage(content="", response_metadata={"stop_reason": "refusal"})
        with pytest.raises(ChatModelError):
            self.handler.check_stop_reason(msg)

    def test_context_window_exceeded_raises(self) -> None:
        msg = AIMessage(
            content="",
            response_metadata={"stop_reason": "model_context_window_exceeded"},
        )
        with pytest.raises(ChatModelError):
            self.handler.check_stop_reason(msg)

    def test_no_stop_reason_no_raise(self) -> None:
        msg = AIMessage(content="ok", response_metadata={})
        self.handler.check_stop_reason(msg)

    def test_converse_camel_case_key_ignored(self) -> None:
        """Invoke handler must not react to stopReason (camelCase)."""
        msg = AIMessage(content="", response_metadata={"stopReason": "max_tokens"})
        self.handler.check_stop_reason(msg)  # should not raise


# ---------------------------------------------------------------------------
# BedrockConversePayloadHandler — check_stop_reason
# ---------------------------------------------------------------------------


class TestBedrockConverseCheckStopReason:
    def setup_method(self) -> None:
        self.handler = BedrockConversePayloadHandler(make_converse_model())  # type: ignore[arg-type]

    def test_end_turn_no_raise(self) -> None:
        msg = AIMessage(content="ok", response_metadata={"stopReason": "end_turn"})
        self.handler.check_stop_reason(msg)

    def test_max_tokens_raises(self) -> None:
        msg = AIMessage(content="", response_metadata={"stopReason": "max_tokens"})
        with pytest.raises(ChatModelError):
            self.handler.check_stop_reason(msg)

    def test_guardrail_intervened_raises(self) -> None:
        msg = AIMessage(
            content="", response_metadata={"stopReason": "guardrail_intervened"}
        )
        with pytest.raises(ChatModelError):
            self.handler.check_stop_reason(msg)

    def test_content_filtered_raises(self) -> None:
        msg = AIMessage(
            content="", response_metadata={"stopReason": "content_filtered"}
        )
        with pytest.raises(ChatModelError):
            self.handler.check_stop_reason(msg)

    def test_context_window_exceeded_raises(self) -> None:
        msg = AIMessage(
            content="",
            response_metadata={"stopReason": "model_context_window_exceeded"},
        )
        with pytest.raises(ChatModelError):
            self.handler.check_stop_reason(msg)

    def test_no_stop_reason_no_raise(self) -> None:
        msg = AIMessage(content="ok", response_metadata={})
        self.handler.check_stop_reason(msg)

    def test_invoke_snake_case_key_ignored(self) -> None:
        """Converse handler must not react to stop_reason (snake_case)."""
        msg = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
        self.handler.check_stop_reason(msg)  # should not raise


# ---------------------------------------------------------------------------
# Null-safety: thinking_enabled detection
# ---------------------------------------------------------------------------


class TestBedrockInvokeThinkingNullSafety:
    """model_kwargs or its nested values may be None — must not raise."""

    def test_model_kwargs_is_none(self) -> None:
        model = type("FakeChatBedrock", (), {})()
        model.model_kwargs = None
        handler = BedrockInvokePayloadHandler(model)
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "any"

    def test_thinking_value_is_none(self) -> None:
        model = type("FakeChatBedrock", (), {})()
        model.model_kwargs = {"thinking": None}
        handler = BedrockInvokePayloadHandler(model)
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "any"

    def test_model_kwargs_attribute_missing(self) -> None:
        model = type("FakeChatBedrock", (), {})()
        handler = BedrockInvokePayloadHandler(model)
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "any"


class TestBedrockConverseThinkingNullSafety:
    """additional_model_request_fields or its nested values may be None — must not raise."""

    def test_additional_fields_is_none(self) -> None:
        model = type("FakeChatBedrockConverse", (), {})()
        model.additional_model_request_fields = None
        handler = BedrockConversePayloadHandler(model)
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "any"

    def test_thinking_value_is_none(self) -> None:
        model = type("FakeChatBedrockConverse", (), {})()
        model.additional_model_request_fields = {"thinking": None}
        handler = BedrockConversePayloadHandler(model)
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "any"

    def test_additional_fields_attribute_missing(self) -> None:
        model = type("FakeChatBedrockConverse", (), {})()
        handler = BedrockConversePayloadHandler(model)
        result = handler.get_tool_binding_kwargs([], "any")
        assert result["tool_choice"] == "any"
