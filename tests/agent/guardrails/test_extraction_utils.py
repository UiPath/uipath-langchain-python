"""Tests for guardrail utility functions."""

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.guardrails.utils import (
    _extract_tool_args_from_message,
    _extract_tool_output_data,
    _extract_tools_args_from_message,
    _tool_call_field,
    current_tool_call_args,
    get_message_content,
)
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState


class TestExtractToolArgsFromMessage:
    """Tests for _extract_tool_args_from_message function."""

    def test_extracts_args_from_matching_tool(self):
        """Should extract args from matching tool call."""
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"param1": "value1", "param2": 123},
                    "id": "call_1",
                }
            ],
        )
        result = _extract_tool_args_from_message(message, "test_tool")
        assert result == {"param1": "value1", "param2": 123}

    def test_returns_empty_dict_for_non_matching_tool(self):
        """Should return empty dict when tool name doesn't match."""
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "other_tool", "args": {"data": "value"}, "id": "call_1"}
            ],
        )
        result = _extract_tool_args_from_message(message, "test_tool")
        assert result == {}

    def test_returns_empty_dict_for_non_ai_message(self):
        """Should return empty dict when message is not AIMessage."""
        message = HumanMessage(content="Test message")
        result = _extract_tool_args_from_message(message, "test_tool")
        assert result == {}

    def test_returns_first_matching_tool_when_multiple(self):
        """Should return args from first matching tool call."""
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "test_tool", "args": {"first": "call"}, "id": "call_1"},
                {"name": "test_tool", "args": {"second": "call"}, "id": "call_2"},
            ],
        )
        result = _extract_tool_args_from_message(message, "test_tool")
        assert result == {"first": "call"}


class TestExtractToolsArgsFromMessage:
    """Tests for _extract_tools_args_from_message function."""

    def test_extracts_args_from_all_tool_calls(self):
        """Should extract args from all tool calls."""
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool1", "args": {"arg1": "val1"}, "id": "call_1"},
                {"name": "tool2", "args": {"arg2": "val2"}, "id": "call_2"},
                {"name": "tool3", "args": {"arg3": "val3"}, "id": "call_3"},
            ],
        )
        result = _extract_tools_args_from_message(message)
        assert result == [{"arg1": "val1"}, {"arg2": "val2"}, {"arg3": "val3"}]

    def test_returns_empty_list_for_non_ai_message(self):
        """Should return empty list when message is not AIMessage."""
        message = HumanMessage(content="Test message")
        result = _extract_tools_args_from_message(message)
        assert result == []

    def test_returns_empty_list_when_no_tool_calls(self):
        """Should return empty list when AIMessage has no tool calls."""
        message = AIMessage(content="Test response")
        result = _extract_tools_args_from_message(message)
        assert result == []


class TestExtractToolOutputData:
    """Tests for _extract_tool_output_data function."""

    def test_extracts_json_dict_content(self):
        """Should parse and return dict when content is JSON string."""
        json_content = json.dumps({"result": "success", "data": {"value": 42}})
        state = AgentGuardrailsGraphState(
            messages=[ToolMessage(content=json_content, tool_call_id="call_1")]
        )
        result = _extract_tool_output_data(state)
        assert result == {"result": "success", "data": {"value": 42}}

    def test_wraps_non_json_string_in_output_field(self):
        """Should wrap non-JSON string content in 'output' field."""
        state = AgentGuardrailsGraphState(
            messages=[ToolMessage(content="Plain text result", tool_call_id="call_1")]
        )
        result = _extract_tool_output_data(state)
        assert result == {"output": "Plain text result"}

    def test_returns_empty_dict_for_empty_messages(self):
        """Should return empty dict when state has no messages."""
        state = AgentGuardrailsGraphState(messages=[])
        result = _extract_tool_output_data(state)
        assert result == {}

    def test_returns_empty_dict_for_non_tool_message(self):
        """Should return empty dict when last message is not ToolMessage."""
        state = AgentGuardrailsGraphState(
            messages=[AIMessage(content="Not a tool message")]
        )
        result = _extract_tool_output_data(state)
        assert result == {}


class TestGetMessageContent:
    """Tests for get_message_content function."""

    def test_extracts_string_content_from_human_message(self):
        """Should extract string content from HumanMessage."""
        message = HumanMessage(content="Hello from human")
        result = get_message_content(message)
        assert result == "Hello from human"

    def test_extracts_content_from_ai_message(self):
        """Should extract content from AIMessage."""
        message = AIMessage(content="AI response")
        result = get_message_content(message)
        assert result == "AI response"

    def test_extracts_content_from_tool_message(self):
        """Should extract content from ToolMessage."""
        message = ToolMessage(content="Tool result", tool_call_id="call_1")
        result = get_message_content(message)
        assert result == "Tool result"

    def test_handles_empty_content(self):
        """Should handle empty content string."""
        message = AIMessage(content="")
        result = get_message_content(message)
        assert result == ""


def _two_calls() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "test_tool", "args": {"file": "a"}, "id": "call_1"},
            {"name": "test_tool", "args": {"file": "b"}, "id": "call_2"},
        ],
    )


class TestCurrentToolCallArgs:
    """Tests for current_tool_call_args function."""

    def test_pre_execution_selects_the_first_call_without_a_result(self):
        messages: list[AnyMessage] = [
            _two_calls(),
            ToolMessage(content="a done", tool_call_id="call_1"),
        ]

        result = current_tool_call_args(
            messages, "test_tool", ExecutionStage.PRE_EXECUTION
        )

        assert result == {"file": "b"}

    def test_post_execution_selects_the_call_the_last_result_answers(self):
        messages: list[AnyMessage] = [
            _two_calls(),
            ToolMessage(content="a done", tool_call_id="call_1"),
            ToolMessage(content="b done", tool_call_id="call_2"),
        ]

        result = current_tool_call_args(
            messages, "test_tool", ExecutionStage.POST_EXECUTION
        )

        assert result == {"file": "b"}

    def test_no_messages_returns_empty(self):
        assert (
            current_tool_call_args([], "test_tool", ExecutionStage.PRE_EXECUTION) == {}
        )

    def test_no_tool_calls_returns_empty(self):
        messages: list[AnyMessage] = [
            HumanMessage(content="hi"),
            AIMessage(content="no tools"),
        ]

        result = current_tool_call_args(
            messages, "test_tool", ExecutionStage.PRE_EXECUTION
        )

        assert result == {}

    def test_pre_execution_without_tool_message_bookkeeping_falls_back_to_first_match(
        self,
    ):
        messages: list[AnyMessage] = [
            _two_calls(),
            HumanMessage(content="interjection"),
        ]

        result = current_tool_call_args(
            messages, "test_tool", ExecutionStage.PRE_EXECUTION
        )

        assert result == {"file": "a"}

    def test_post_execution_without_a_matching_result_falls_back_to_first_match(self):
        messages: list[AnyMessage] = [
            _two_calls(),
            ToolMessage(content="other", tool_call_id="call_unknown"),
        ]

        result = current_tool_call_args(
            messages, "test_tool", ExecutionStage.POST_EXECUTION
        )

        assert result == {"file": "a"}

    def test_post_execution_ignores_a_result_for_another_tool(self):
        messages: list[AnyMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_a", "args": {"x": 1}, "id": "call_1"},
                    {"name": "tool_b", "args": {"y": 2}, "id": "call_2"},
                ],
            ),
            ToolMessage(content="b done", tool_call_id="call_2"),
        ]

        result = current_tool_call_args(
            messages, "tool_a", ExecutionStage.POST_EXECUTION
        )

        assert result == {"x": 1}

    def test_tool_call_field_reads_dicts_and_objects(self):
        assert _tool_call_field({"id": "call_1"}, "id") == "call_1"
        assert _tool_call_field(SimpleNamespace(id="call_2"), "id") == "call_2"
        assert _tool_call_field(SimpleNamespace(), "id") is None
