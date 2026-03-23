"""Tests for get_tool_binding_kwargs across all payload handlers."""

from unittest.mock import Mock

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from uipath_langchain.chat.handlers.anthropic import AnthropicPayloadHandler
from uipath_langchain.chat.handlers.base import DefaultModelPayloadHandler
from uipath_langchain.chat.handlers.bedrock import (
    BedrockConversePayloadHandler,
    BedrockInvokePayloadHandler,
)
from uipath_langchain.chat.handlers.gemini import GeminiPayloadHandler
from uipath_langchain.chat.handlers.openai import OpenAIPayloadHandler


def _make_tools(*names: str) -> list[Mock]:
    """Create a list of mock tools with the given names."""
    tools = []
    for name in names:
        tool = Mock(spec=BaseTool)
        tool.name = name
        tools.append(tool)
    return tools


# ---------------------------------------------------------------------------
# Default handler
# ---------------------------------------------------------------------------


class TestDefaultGetToolBindingKwargs:
    """DefaultModelPayloadHandler returns only tool_choice."""

    def setup_method(self):
        self.handler = DefaultModelPayloadHandler(Mock(spec=BaseChatModel))
        self.tools = _make_tools("tool_a")

    def test_tool_choice_auto(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto"
        )
        assert result == {"tool_choice": "auto"}

    def test_tool_choice_any(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any"
        )
        assert result == {"tool_choice": "any"}

    def test_extra_params_not_leaked(self):
        """parallel_tool_calls and strict_mode must not appear in the result."""
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            strict_mode=True,
        )
        assert list(result.keys()) == ["tool_choice"]


# ---------------------------------------------------------------------------
# OpenAI handler
# ---------------------------------------------------------------------------


class TestOpenAIGetToolBindingKwargs:
    """OpenAIPayloadHandler returns tool_choice, parallel_tool_calls, strict."""

    def setup_method(self):
        self.handler = OpenAIPayloadHandler(Mock(spec=BaseChatModel))
        self.tools = _make_tools("tool_a")

    def test_tool_choice_auto(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto"
        )
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any"
        )
        assert result["tool_choice"] == "any"

    def test_parallel_tool_calls_true(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", parallel_tool_calls=True
        )
        assert result["parallel_tool_calls"] is True

    def test_parallel_tool_calls_false(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", parallel_tool_calls=False
        )
        assert result["parallel_tool_calls"] is False

    def test_strict_mode_true(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", strict_mode=True
        )
        assert result["strict"] is True

    def test_strict_mode_false(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", strict_mode=False
        )
        assert result["strict"] is False

    def test_all_keys_present(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools,
            tool_choice="any",
            parallel_tool_calls=False,
            strict_mode=True,
        )
        assert set(result.keys()) == {"tool_choice", "parallel_tool_calls", "strict"}


# ---------------------------------------------------------------------------
# Anthropic handler
# ---------------------------------------------------------------------------


class TestAnthropicGetToolBindingKwargs:
    """AnthropicPayloadHandler returns tool_choice, parallel_tool_calls, strict."""

    def setup_method(self):
        self.handler = AnthropicPayloadHandler(Mock(spec=BaseChatModel))
        self.tools = _make_tools("tool_a")

    def test_tool_choice_auto(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto"
        )
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any"
        )
        assert result["tool_choice"] == "any"

    def test_parallel_tool_calls_true(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", parallel_tool_calls=True
        )
        assert result["parallel_tool_calls"] is True

    def test_parallel_tool_calls_false(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", parallel_tool_calls=False
        )
        assert result["parallel_tool_calls"] is False

    def test_strict_mode_true(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", strict_mode=True
        )
        assert result["strict"] is True

    def test_strict_mode_false(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", strict_mode=False
        )
        assert result["strict"] is False

    def test_all_keys_present(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools,
            tool_choice="any",
            parallel_tool_calls=True,
            strict_mode=False,
        )
        assert set(result.keys()) == {"tool_choice", "parallel_tool_calls", "strict"}


# ---------------------------------------------------------------------------
# Gemini handler
# ---------------------------------------------------------------------------


class TestGeminiGetToolBindingKwargs:
    """GeminiPayloadHandler returns a nested tool_config dict."""

    def setup_method(self):
        self.handler = GeminiPayloadHandler(Mock(spec=BaseChatModel))
        self.tools = _make_tools("get_weather", "search")

    def test_mode_auto(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto"
        )
        config = result["tool_config"]["function_calling_config"]
        assert config["mode"] == "AUTO"

    def test_mode_any(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any"
        )
        config = result["tool_config"]["function_calling_config"]
        assert config["mode"] == "ANY"

    def test_strict_mode_overrides_to_validated(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", strict_mode=True
        )
        config = result["tool_config"]["function_calling_config"]
        assert config["mode"] == "VALIDATED"

    def test_strict_mode_overrides_any_to_validated(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any", strict_mode=True
        )
        config = result["tool_config"]["function_calling_config"]
        assert config["mode"] == "VALIDATED"

    def test_only_tool_config_key(self):
        """parallel_tool_calls and strict do not leak as top-level keys."""
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            strict_mode=False,
        )
        assert list(result.keys()) == ["tool_config"]


# ---------------------------------------------------------------------------
# Bedrock Invoke handler (ChatBedrock)
# ---------------------------------------------------------------------------


class TestBedrockInvokeGetToolBindingKwargs:
    """BedrockInvokePayloadHandler: tool_choice + optional disable_parallel_tool_use."""

    def setup_method(self):
        self.handler = BedrockInvokePayloadHandler(Mock(spec=BaseChatModel))
        self.tools = _make_tools("tool_a")

    def test_tool_choice_auto(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto"
        )
        assert result == {"tool_choice": "auto"}

    def test_tool_choice_any(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any"
        )
        assert result == {"tool_choice": "any"}

    def test_result_contains_tool_choice_key(self):
        """Regression: previously returned empty dict, losing tool_choice."""
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any"
        )
        assert "tool_choice" in result

    def test_parallel_tool_calls_not_included(self):
        """Invoke API uses disable_parallel_tool_use, not parallel_tool_calls."""
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", parallel_tool_calls=True
        )
        assert "parallel_tool_calls" not in result

    def test_strict_mode_not_included(self):
        """Invoke API does not support strict mode."""
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", strict_mode=True
        )
        assert "strict" not in result

    def test_only_tool_choice_returned_when_parallel_true(self):
        """With parallel_tool_calls=True no extra keys are added."""
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools,
            tool_choice="any",
            parallel_tool_calls=True,
            strict_mode=True,
        )
        assert list(result.keys()) == ["tool_choice"]


# ---------------------------------------------------------------------------
# Bedrock Converse handler (ChatBedrockConverse)
# ---------------------------------------------------------------------------


class TestBedrockConverseGetToolBindingKwargs:
    """BedrockConversePayloadHandler: tool_choice + optional strict."""

    def setup_method(self):
        self.handler = BedrockConversePayloadHandler(Mock(spec=BaseChatModel))
        self.tools = _make_tools("tool_a")

    def test_tool_choice_auto(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto"
        )
        assert result == {"tool_choice": "auto"}

    def test_tool_choice_any(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="any"
        )
        assert result == {"tool_choice": "any"}

    def test_strict_mode_included(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", strict_mode=True
        )
        assert result == {"tool_choice": "auto", "strict": True}

    def test_parallel_tool_calls_not_included(self):
        """Converse API does not support parallel_tool_calls."""
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools, tool_choice="auto", parallel_tool_calls=False
        )
        assert "parallel_tool_calls" not in result
        assert "disable_parallel_tool_use" not in result

    def test_only_tool_choice_returned_when_strict_false(self):
        result = self.handler.get_tool_binding_kwargs(
            tools=self.tools,
            tool_choice="any",
            parallel_tool_calls=False,
            strict_mode=False,
        )
        assert list(result.keys()) == ["tool_choice"]
