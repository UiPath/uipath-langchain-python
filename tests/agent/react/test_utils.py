"""Tests for ReAct agent utilities."""

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.utils import (
    count_consecutive_thinking_messages,
    extract_current_tool_call_index,
    find_latest_ai_message,
)


class TestCountSuccessiveCompletions:
    """Test successive completions calculation from message history."""

    def test_empty_messages(self):
        """Should return 0 for empty message list."""
        assert count_consecutive_thinking_messages([]) == 0

    def test_no_ai_messages(self):
        """Should return 0 when no AI messages exist."""
        messages = [HumanMessage(content="test")]
        assert count_consecutive_thinking_messages(messages) == 0

    def test_last_message_not_ai(self):
        """Should return 0 when last message is not AI."""
        messages = [
            AIMessage(content="response"),
            HumanMessage(content="follow-up"),
        ]
        assert count_consecutive_thinking_messages(messages) == 0

    def test_ai_message_with_tool_calls(self):
        """Should return 0 when last AI message has tool calls."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tool",
                tool_calls=[{"name": "test", "args": {}, "id": "call_1"}],
            ),
        ]
        assert count_consecutive_thinking_messages(messages) == 0

    def test_ai_message_without_content(self):
        """Should return 0 when last AI message has no content."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content=""),
        ]
        assert count_consecutive_thinking_messages(messages) == 0

    def test_single_text_completion(self):
        """Should count single text-only AI message."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking"),
        ]
        assert count_consecutive_thinking_messages(messages) == 1

    def test_two_successive_completions(self):
        """Should count multiple consecutive text-only AI messages."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking 1"),
            AIMessage(content="thinking 2"),
        ]
        assert count_consecutive_thinking_messages(messages) == 2

    def test_three_successive_completions(self):
        """Should count all consecutive text-only AI messages at end."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking 1"),
            AIMessage(content="thinking 2"),
            AIMessage(content="thinking 3"),
        ]
        assert count_consecutive_thinking_messages(messages) == 3

    def test_tool_call_resets_count(self):
        """Should only count completions after last tool call."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking 1"),
            AIMessage(
                content="using tool",
                tool_calls=[{"name": "test", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(content="result", tool_call_id="call_1"),
            AIMessage(content="thinking 2"),
            AIMessage(content="thinking 3"),
        ]
        assert count_consecutive_thinking_messages(messages) == 2

    def test_mixed_message_types(self):
        """Should handle complex message patterns correctly."""
        messages = [
            HumanMessage(content="initial query"),
            AIMessage(content="first thought"),
            AIMessage(
                content="calling tool",
                tool_calls=[{"name": "tool1", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(content="tool result", tool_call_id="call_1"),
            AIMessage(content="analyzing result"),
            HumanMessage(content="user follow-up"),
            AIMessage(content="responding to follow-up"),
        ]
        assert count_consecutive_thinking_messages(messages) == 1

    def test_multiple_tool_calls_in_message(self):
        """Should reset count even with multiple tool calls."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking"),
            AIMessage(
                content="using tools",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                    {"name": "tool2", "args": {}, "id": "call_2"},
                ],
            ),
        ]
        assert count_consecutive_thinking_messages(messages) == 0

    def test_ai_message_with_empty_tool_calls_list(self):
        """Should handle AI message with empty tool_calls list."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking", tool_calls=[]),
        ]
        assert count_consecutive_thinking_messages(messages) == 1

    def test_only_ai_messages_all_text(self):
        """Should count all AI messages when all are text-only."""
        messages = [
            AIMessage(content="thought 1"),
            AIMessage(content="thought 2"),
            AIMessage(content="thought 3"),
        ]
        assert count_consecutive_thinking_messages(messages) == 3


class TestFindLatestAiMessage:
    """Test finding the latest AI message from message history."""

    def test_empty_messages(self):
        """Should return None for empty message list."""
        assert find_latest_ai_message([]) is None

    def test_no_ai_messages(self):
        """Should return None when no AI messages exist."""
        messages: list[AnyMessage] = [
            HumanMessage(content="test"),
            ToolMessage(content="result", tool_call_id="call_1"),
        ]
        assert find_latest_ai_message(messages) is None

    def test_single_ai_message(self):
        """Should return the only AI message."""
        ai_message = AIMessage(content="response")
        messages: list[AnyMessage] = [HumanMessage(content="query"), ai_message]

        result = find_latest_ai_message(messages)
        assert result is ai_message

    def test_multiple_ai_messages(self):
        """Should return the latest AI message."""
        first_ai = AIMessage(content="first response")
        second_ai = AIMessage(content="second response")
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            first_ai,
            HumanMessage(content="follow-up"),
            second_ai,
        ]

        result = find_latest_ai_message(messages)
        assert result is second_ai

    def test_ai_message_followed_by_tool_messages(self):
        """Should find AI message even when followed by tool messages."""
        ai_message = AIMessage(
            content="using tools",
            tool_calls=[
                {"name": "tool1", "args": {}, "id": "call_1"},
                {"name": "tool2", "args": {}, "id": "call_2"},
            ],
        )
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            ai_message,
            ToolMessage(content="result1", tool_call_id="call_1"),
            ToolMessage(content="result2", tool_call_id="call_2"),
        ]

        result = find_latest_ai_message(messages)
        assert result is ai_message


class TestExtractCurrentToolCallIndex:
    """Test extracting current tool call index from message history."""

    def test_empty_messages(self):
        """Should raise AgentRuntimeError for empty message list."""
        with pytest.raises(AgentRuntimeError) as exc_info:
            extract_current_tool_call_index([])

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.STATE_ERROR
        )

    def test_no_ai_messages(self):
        """Should raise AgentRuntimeError when no AI messages exist."""
        messages: list[AnyMessage] = [HumanMessage(content="test")]
        with pytest.raises(AgentRuntimeError) as exc_info:
            extract_current_tool_call_index(messages)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.STATE_ERROR
        )

    def test_ai_message_no_tool_calls(self):
        """Should raise AgentRuntimeError when AI message has no tool calls."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(content="response"),
        ]
        with pytest.raises(AgentRuntimeError) as exc_info:
            extract_current_tool_call_index(messages)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.STATE_ERROR
        )

    def test_single_tool_call_not_executed(self):
        """Should return 0 for first tool call when none executed."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tool",
                tool_calls=[{"name": "test", "args": {}, "id": "call_1"}],
            ),
        ]
        assert extract_current_tool_call_index(messages) == 0

    def test_single_tool_call_executed(self):
        """Should return None when single tool call is already executed."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tool",
                tool_calls=[{"name": "test", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(content="result", tool_call_id="call_1"),
        ]
        assert extract_current_tool_call_index(messages) is None

    def test_multiple_tool_calls_none_executed(self):
        """Should return 0 for first tool call when none executed."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tools",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                    {"name": "tool2", "args": {}, "id": "call_2"},
                    {"name": "tool3", "args": {}, "id": "call_3"},
                ],
            ),
        ]
        assert extract_current_tool_call_index(messages) == 0

    def test_multiple_tool_calls_first_executed(self):
        """Should return 1 for second tool call when first is executed."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tools",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                    {"name": "tool2", "args": {}, "id": "call_2"},
                    {"name": "tool3", "args": {}, "id": "call_3"},
                ],
            ),
            ToolMessage(content="result1", tool_call_id="call_1"),
        ]
        assert extract_current_tool_call_index(messages) == 1

    def test_multiple_tool_calls_some_executed(self):
        """Should return correct index when some tool calls are executed."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tools",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                    {"name": "tool2", "args": {}, "id": "call_2"},
                    {"name": "tool3", "args": {}, "id": "call_3"},
                ],
            ),
            ToolMessage(content="result1", tool_call_id="call_1"),
            ToolMessage(content="result2", tool_call_id="call_2"),
        ]
        assert extract_current_tool_call_index(messages) == 2

    def test_multiple_tool_calls_all_executed(self):
        """Should return None when all tool calls are executed."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tools",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                    {"name": "tool2", "args": {}, "id": "call_2"},
                ],
            ),
            ToolMessage(content="result1", tool_call_id="call_1"),
            ToolMessage(content="result2", tool_call_id="call_2"),
        ]
        assert extract_current_tool_call_index(messages) is None

    def test_tool_name_filtering(self):
        """Should filter tool calls by name when tool_name is provided."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tools",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                    {"name": "tool2", "args": {}, "id": "call_2"},
                    {"name": "tool1", "args": {}, "id": "call_3"},
                ],
            ),
            ToolMessage(content="result1", tool_call_id="call_1"),
        ]
        # Should find the second tool1 call at index 2
        assert extract_current_tool_call_index(messages, "tool1") == 2
        # Should find the first tool2 call at index 1
        assert extract_current_tool_call_index(messages, "tool2") == 1

    def test_tool_name_filtering_all_executed(self):
        """Should return None when all matching tool calls are executed."""
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tools",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                    {"name": "tool2", "args": {}, "id": "call_2"},
                    {"name": "tool1", "args": {}, "id": "call_3"},
                ],
            ),
            ToolMessage(content="result1", tool_call_id="call_1"),
            ToolMessage(content="result2", tool_call_id="call_2"),
            ToolMessage(content="result3", tool_call_id="call_3"),
        ]
        assert extract_current_tool_call_index(messages, "tool1") is None
        assert extract_current_tool_call_index(messages, "tool2") is None

    def test_mixed_message_types(self):
        """Should handle complex message patterns correctly."""
        messages: list[AnyMessage] = [
            HumanMessage(content="initial query"),
            AIMessage(
                content="first ai response",
                tool_calls=[
                    {"name": "tool1", "args": {}, "id": "call_1"},
                ],
            ),
            ToolMessage(content="tool1 result", tool_call_id="call_1"),
            HumanMessage(content="follow-up"),
            AIMessage(
                content="second ai response",
                tool_calls=[
                    {"name": "tool2", "args": {}, "id": "call_2"},
                    {"name": "tool3", "args": {}, "id": "call_3"},
                ],
            ),
            ToolMessage(content="tool2 result", tool_call_id="call_2"),
        ]
        # Should find the latest AI message and return index 1 (tool3)
        assert extract_current_tool_call_index(messages) == 1

    def test_unexpected_message_type(self):
        """Should raise AgentRuntimeError for unexpected message types."""
        from unittest.mock import Mock

        # Create a mock message that's not AI, Tool, or Human
        unexpected_message = Mock()
        unexpected_message.__class__.__name__ = "UnexpectedMessage"

        messages: list[AnyMessage] = [unexpected_message]

        with pytest.raises(AgentRuntimeError) as exc_info:
            extract_current_tool_call_index(messages)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.STATE_ERROR
        )
