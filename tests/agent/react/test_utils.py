"""Tests for ReAct agent utilities."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from uipath_langchain.agent.react.utils import (
    count_successive_completions,
    resolve_input_model,
    resolve_output_model,
)


class TestCountSuccessiveCompletions:
    """Test successive completions calculation from message history."""

    def test_empty_messages(self):
        """Should return 0 for empty message list."""
        assert count_successive_completions([]) == 0

    def test_no_ai_messages(self):
        """Should return 0 when no AI messages exist."""
        messages = [HumanMessage(content="test")]
        assert count_successive_completions(messages) == 0

    def test_last_message_not_ai(self):
        """Should return 0 when last message is not AI."""
        messages = [
            AIMessage(content="response"),
            HumanMessage(content="follow-up"),
        ]
        assert count_successive_completions(messages) == 0

    def test_ai_message_with_tool_calls(self):
        """Should return 0 when last AI message has tool calls."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tool",
                tool_calls=[{"name": "test", "args": {}, "id": "call_1"}],
            ),
        ]
        assert count_successive_completions(messages) == 0

    def test_ai_message_without_content(self):
        """Should return 0 when last AI message has no content."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content=""),
        ]
        assert count_successive_completions(messages) == 0

    def test_single_text_completion(self):
        """Should count single text-only AI message."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking"),
        ]
        assert count_successive_completions(messages) == 1

    def test_two_successive_completions(self):
        """Should count multiple consecutive text-only AI messages."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking 1"),
            AIMessage(content="thinking 2"),
        ]
        assert count_successive_completions(messages) == 2

    def test_three_successive_completions(self):
        """Should count all consecutive text-only AI messages at end."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking 1"),
            AIMessage(content="thinking 2"),
            AIMessage(content="thinking 3"),
        ]
        assert count_successive_completions(messages) == 3

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
        assert count_successive_completions(messages) == 2

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
        assert count_successive_completions(messages) == 1

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
        assert count_successive_completions(messages) == 0

    def test_ai_message_with_empty_tool_calls_list(self):
        """Should handle AI message with empty tool_calls list."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="thinking", tool_calls=[]),
        ]
        assert count_successive_completions(messages) == 1

    def test_only_ai_messages_all_text(self):
        """Should count all AI messages when all are text-only."""
        messages = [
            AIMessage(content="thought 1"),
            AIMessage(content="thought 2"),
            AIMessage(content="thought 3"),
        ]
        assert count_successive_completions(messages) == 3


class TestResolveInputModel:
    """Test input model resolution from JSON schema."""

    def test_returns_base_model_when_schema_is_none(self):
        """Should return BaseModel when no schema provided."""
        result = resolve_input_model(None)
        assert result is BaseModel

    def test_creates_model_from_simple_schema(self):
        """Should create Pydantic model from simple JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = resolve_input_model(schema)
        assert issubclass(result, BaseModel)
        assert "name" in result.model_fields

    def test_creates_model_from_nested_schema(self):
        """Should create Pydantic model from nested JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                    },
                },
            },
        }
        result = resolve_input_model(schema)
        assert issubclass(result, BaseModel)


class TestResolveOutputModel:
    """Test output model resolution from JSON schema."""

    def test_returns_end_execution_schema_when_none(self):
        """Should return END_EXECUTION_TOOL args_schema when no schema provided."""
        from uipath.agent.react import END_EXECUTION_TOOL

        result = resolve_output_model(None)
        assert result is END_EXECUTION_TOOL.args_schema

    def test_creates_model_from_output_schema(self):
        """Should create Pydantic model from output JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "success": {"type": "boolean"},
            },
            "required": ["result"],
        }
        result = resolve_output_model(schema)
        assert issubclass(result, BaseModel)
        assert "result" in result.model_fields
