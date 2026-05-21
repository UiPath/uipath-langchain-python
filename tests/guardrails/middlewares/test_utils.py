"""Tests for guardrail middleware utility functions."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from uipath_langchain.guardrails.middlewares._utils import (
    create_modified_tool_request,
    create_modified_tool_result,
    extract_text_from_messages,
)


def _make_request(args: dict[str, Any] | str | None = None) -> ToolCallRequest:
    resolved_args: dict[str, Any] | str = args if args is not None else {"x": 1}
    return ToolCallRequest(
        tool_call={"id": "tc1", "name": "my_tool", "args": resolved_args},
        tool=MagicMock(),
        state={},
        runtime=MagicMock(),
    )


def _make_tool_message(content: str = '{"result": "ok"}') -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="tc1")


# ---------------------------------------------------------------------------
# TestCreateModifiedToolRequest
# ---------------------------------------------------------------------------


class TestCreateModifiedToolRequest:
    def test_happy_path_replaces_args(self) -> None:
        request = _make_request(args={"x": 1})
        result = create_modified_tool_request(request, {"x": 99})
        assert result.tool_call["args"] == {"x": 99}

    def test_fallback_when_replace_raises_type_error(self) -> None:
        request = _make_request(args={"x": 1})
        with patch("dataclasses.replace", side_effect=TypeError("not a dataclass")):
            result = create_modified_tool_request(request, {"x": 2})
        assert result.tool_call["args"] == {"x": 2}

    def test_str_args_accepted(self) -> None:
        request = _make_request(args={"x": 1})
        result = create_modified_tool_request(request, "filtered string")
        assert result.tool_call["args"] == "filtered string"


# ---------------------------------------------------------------------------
# TestCreateModifiedToolResult
# ---------------------------------------------------------------------------


class TestCreateModifiedToolResult:
    def test_tool_message_single_output_key_str_unwrapped(self) -> None:
        msg = _make_tool_message("original text")
        result = create_modified_tool_result(msg, {"output": "filtered"})
        assert isinstance(result, ToolMessage)
        assert result.content == "filtered"

    def test_tool_message_multi_key_dict_json_serialized(self) -> None:
        msg = _make_tool_message("original text")
        result = create_modified_tool_result(msg, {"a": 1, "b": 2})
        assert isinstance(result, ToolMessage)
        assert result.content == '{"a": 1, "b": 2}'

    def test_tool_message_str_output_used_directly(self) -> None:
        msg = _make_tool_message("original text")
        result = create_modified_tool_result(msg, "plain string")
        assert isinstance(result, ToolMessage)
        assert result.content == "plain string"

    def test_command_with_tool_message_updates_content(self) -> None:
        tool_msg = _make_tool_message('{"x": 1}')
        cmd: Command[Any] = Command(update={"messages": [tool_msg]})
        result = create_modified_tool_result(cmd, {"x": "filtered"})
        assert isinstance(result, Command)
        inner_msg = result.update["messages"][0]  # type: ignore[index]
        assert isinstance(inner_msg, ToolMessage)
        assert "filtered" in inner_msg.content

    def test_command_without_tool_message_returned_unchanged(self) -> None:
        cmd: Command[Any] = Command(update={"messages": [AIMessage(content="hi")]})
        result = create_modified_tool_result(cmd, {"output": "new"})
        assert result is cmd

    def test_unknown_type_returned_unchanged(self) -> None:
        obj = "not a message"
        result = create_modified_tool_result(obj, {"output": "new"})  # type: ignore[arg-type]
        assert result is obj


# ---------------------------------------------------------------------------
# TestExtractTextFromMessages
# ---------------------------------------------------------------------------


class TestExtractTextFromMessages:
    def test_human_message_str_content(self) -> None:
        msgs = [HumanMessage(content="hello")]
        assert extract_text_from_messages(msgs) == "hello"

    def test_multimodal_list_extracts_text_part(self) -> None:
        msgs = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                ]
            )
        ]
        assert extract_text_from_messages(msgs) == "hi"

    def test_non_text_parts_ignored(self) -> None:
        msgs = [
            HumanMessage(
                content=[{"type": "image_url", "url": "http://example.com/img.png"}]
            )
        ]
        assert extract_text_from_messages(msgs) == ""

    def test_tool_messages_skipped(self) -> None:
        msgs = [ToolMessage(content="tool output", tool_call_id="tc1")]
        assert extract_text_from_messages(msgs) == ""

    def test_multiple_messages_joined_with_newline(self) -> None:
        msgs = [HumanMessage(content="first"), AIMessage(content="second")]
        assert extract_text_from_messages(msgs) == "first\nsecond"
