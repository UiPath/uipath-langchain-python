"""Tests for the forced-extraction helpers."""

import logging
from typing import Any, TypeVar, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.content import create_tool_call
from pydantic import BaseModel, ConfigDict

from uipath_langchain.agent.react.forced_extraction import (
    _ensure_trailing_user_turn,
    _strip_reasoning_blocks,
)
from uipath_langchain.chat.thinking import strip_thinking

_ModelT = TypeVar("_ModelT")


def _strip(model: _ModelT) -> _ModelT:
    """Call the duck-typed ``strip_thinking`` with a stand-in model.

    ``strip_thinking`` only reads attributes and ``model_copy``, so the fakes below
    satisfy it at runtime; the casts keep the fake's own type on the way out.
    """
    return cast(_ModelT, strip_thinking(cast(BaseChatModel, model)))


class _FakeConverse(BaseModel):
    """Stand-in for ChatBedrockConverse: thinking lives in additional_model_request_fields."""

    model_config = ConfigDict(extra="allow")
    additional_model_request_fields: dict[str, Any] | None = None


class _FakeInvoke(BaseModel):
    """Stand-in for ChatBedrock (Invoke): thinking lives in model_kwargs."""

    model_config = ConfigDict(extra="allow")
    model_kwargs: dict[str, Any] = {}


class _FakeNative(BaseModel):
    """Stand-in for ChatAnthropic: thinking is a top-level attribute."""

    model_config = ConfigDict(extra="allow")
    thinking: dict[str, Any] | None = None


class TestStripThinking:
    """strip_thinking removes reasoning config across transports, keeping the rest."""

    def test_converse_removes_thinking_keeps_others(self) -> None:
        model = _FakeConverse(
            additional_model_request_fields={
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "anthropic_beta": ["x"],
            }
        )
        result = _strip(model)
        assert result.additional_model_request_fields == {"anthropic_beta": ["x"]}

    def test_converse_removes_thinking_and_output_config(self) -> None:
        model = _FakeConverse(
            additional_model_request_fields={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            }
        )
        result = _strip(model)
        assert result.additional_model_request_fields == {}

    def test_invoke_removes_thinking_from_model_kwargs(self) -> None:
        model = _FakeInvoke(
            model_kwargs={"thinking": {"type": "enabled"}, "top_p": 0.9}
        )
        result = _strip(model)
        assert result.model_kwargs == {"top_p": 0.9}

    def test_native_clears_thinking_attribute(self) -> None:
        model = _FakeNative(thinking={"type": "adaptive"})
        result = _strip(model)
        assert result.thinking is None

    def test_no_thinking_returns_same_instance(self) -> None:
        model = _FakeConverse(additional_model_request_fields={"anthropic_beta": ["x"]})
        assert _strip(model) is model

    def test_copy_failure_returns_original_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A model_copy failure degrades to the original model but is logged, not
        swallowed silently — otherwise the extraction call goes out thinking-on."""

        class _CopyFails:
            additional_model_request_fields = {"thinking": {"type": "enabled"}}
            model_kwargs: dict[str, Any] = {}
            thinking = None

            def model_copy(self, update: Any) -> Any:
                raise RuntimeError("copy blew up")

        model = _CopyFails()
        with caplog.at_level(logging.WARNING):
            result = _strip(model)

        assert result is model
        assert any(
            "strip thinking config" in r.getMessage().lower() for r in caplog.records
        )


class TestStripReasoningBlocks:
    """_strip_reasoning_blocks drops reasoning blocks, keeps text and tool calls."""

    def test_keeps_text_drops_reasoning(self) -> None:
        msg = AIMessage(
            content=[
                {"type": "reasoning_content", "reasoning_content": {"text": "think"}},
                {"type": "text", "text": "answer"},
            ]
        )
        out = _strip_reasoning_blocks([msg])
        assert len(out) == 1
        assert out[0].content == [{"type": "text", "text": "answer"}]

    def test_drops_reasoning_only_turn(self) -> None:
        human = HumanMessage(content="q")
        reasoning_only = AIMessage(
            content=[{"type": "reasoning_content", "reasoning_content": {"text": "t"}}]
        )
        out = _strip_reasoning_blocks([human, reasoning_only])
        assert out == [human]

    def test_keeps_turn_with_tool_call_even_if_content_empties(self) -> None:
        msg = AIMessage(
            content=[{"type": "reasoning_content", "reasoning_content": {"text": "t"}}],
            tool_calls=[create_tool_call(name="end_execution", args={}, id="call_1")],
        )
        out = _strip_reasoning_blocks([msg])
        assert len(out) == 1
        kept = out[0]
        assert isinstance(kept, AIMessage)
        assert kept.content == []
        assert kept.tool_calls[0]["name"] == "end_execution"

    def test_strips_redacted_thinking_blocks(self) -> None:
        """redacted_thinking can't be replayed on a thinking-off call either."""
        msg = AIMessage(
            content=[
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "answer"},
            ]
        )
        out = _strip_reasoning_blocks([msg])
        assert out[0].content == [{"type": "text", "text": "answer"}]

    def test_string_content_untouched(self) -> None:
        msg = AIMessage(content="plain answer")
        out = _strip_reasoning_blocks([msg])
        assert out[0] is msg

    def test_non_ai_messages_untouched(self) -> None:
        human = HumanMessage(content="q")
        out = _strip_reasoning_blocks([human])
        assert out == [human]


class TestEnsureTrailingUserTurn:
    """_ensure_trailing_user_turn ends the request on a user turn without doubling up."""

    def test_appends_user_turn_after_assistant(self) -> None:
        """A surviving stalled assistant turn gets a terse, tool-neutral user turn."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="q"),
            AIMessage(content="answer"),
        ]
        out = _ensure_trailing_user_turn(msgs)
        assert len(out) == 3
        assert isinstance(out[-2], AIMessage)
        assert isinstance(out[-1], HumanMessage)
        # a terse, non-empty user turn so the forced call is accepted; wording stays
        # tool-neutral (no "finished"/"result" framing that makes the model reply to it)
        assert isinstance(out[-1].content, str) and out[-1].content
        assert "tool" in out[-1].content.lower()

    def test_no_turn_added_when_already_user(self) -> None:
        """Stripping left a user turn last (reasoning-only stall dropped) — add nothing."""
        msgs: list[AnyMessage] = [HumanMessage(content="the task")]
        out = _ensure_trailing_user_turn(msgs)
        assert out == msgs

    def test_no_turn_added_when_ends_on_tool_result(self) -> None:
        """A trailing tool result is already user-role — don't append a second user turn."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="q"),
            AIMessage(
                content="",
                tool_calls=[create_tool_call(name="t", args={}, id="c1")],
            ),
            ToolMessage(content="result", tool_call_id="c1"),
        ]
        out = _ensure_trailing_user_turn(msgs)
        assert out == msgs

    def test_appends_after_non_user_terminal_message(self) -> None:
        """A trailing non-user-role message (e.g. system) still gets a user turn."""
        msgs: list[AnyMessage] = [SystemMessage(content="sys")]
        out = _ensure_trailing_user_turn(msgs)
        assert len(out) == 2
        assert isinstance(out[-1], HumanMessage)
