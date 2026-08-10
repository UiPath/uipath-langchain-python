"""Tests for the forced-extraction helpers."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.content import create_tool_call
from pydantic import BaseModel, ConfigDict

from uipath_langchain.agent.react.forced_extraction import (
    _strip_reasoning_blocks,
    _with_extraction_nudge,
    _without_thinking,
)


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


class TestWithoutThinking:
    """_without_thinking removes reasoning config across transports, keeping the rest."""

    def test_converse_removes_thinking_keeps_others(self) -> None:
        model = _FakeConverse(
            additional_model_request_fields={
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "anthropic_beta": ["x"],
            }
        )
        result = _without_thinking(model)  # type: ignore[arg-type]
        assert result.additional_model_request_fields == {"anthropic_beta": ["x"]}

    def test_converse_removes_thinking_and_output_config(self) -> None:
        model = _FakeConverse(
            additional_model_request_fields={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            }
        )
        result = _without_thinking(model)  # type: ignore[arg-type]
        assert result.additional_model_request_fields == {}

    def test_invoke_removes_thinking_from_model_kwargs(self) -> None:
        model = _FakeInvoke(
            model_kwargs={"thinking": {"type": "enabled"}, "top_p": 0.9}
        )
        result = _without_thinking(model)  # type: ignore[arg-type]
        assert result.model_kwargs == {"top_p": 0.9}

    def test_native_clears_thinking_attribute(self) -> None:
        model = _FakeNative(thinking={"type": "adaptive"})
        result = _without_thinking(model)  # type: ignore[arg-type]
        assert result.thinking is None

    def test_no_thinking_returns_same_instance(self) -> None:
        model = _FakeConverse(additional_model_request_fields={"anthropic_beta": ["x"]})
        assert _without_thinking(model) is model  # type: ignore[arg-type]


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
        assert out[0].content == []
        assert out[0].tool_calls[0]["name"] == "end_execution"

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


class TestExtractionNudge:
    """_with_extraction_nudge ends on a user turn without creating consecutive user turns."""

    def test_appends_user_turn_after_assistant(self) -> None:
        msgs = [HumanMessage(content="q"), AIMessage(content="answer")]
        out = _with_extraction_nudge(msgs)
        assert isinstance(out[-1], HumanMessage)
        # tool-neutral: mentions continuing, not only end_execution, so a mid-task
        # stall isn't pushed to terminate early
        assert "continue" in out[-1].content
        assert "end_execution" in out[-1].content
        assert isinstance(out[-2], AIMessage)

    def test_merges_into_trailing_user_turn(self) -> None:
        msgs = [HumanMessage(content="the task")]
        out = _with_extraction_nudge(msgs)
        assert len(out) == 1
        assert isinstance(out[-1], HumanMessage)
        assert "the task" in out[-1].content
        assert "end_execution" in out[-1].content

    def test_merges_into_list_content_user_turn(self) -> None:
        msgs = [HumanMessage(content=[{"type": "text", "text": "the task"}])]
        out = _with_extraction_nudge(msgs)
        assert len(out) == 1
        assert out[-1].content[-1]["type"] == "text"
        assert "end_execution" in out[-1].content[-1]["text"]
