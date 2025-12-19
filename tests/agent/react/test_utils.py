"""Tests for ReAct agent utilities."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from uipath_langchain.agent.react.utils import (
    count_consecutive_thinking_messages,
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
