"""Tests for router_utils.py module."""

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from uipath_langchain.agent.exceptions import AgentNodeRoutingException
from uipath_langchain.agent.react.router_utils import validate_last_message_is_AI


class TestValidateLastMessageIsAI:
    """Test cases for validate_last_message_is_AI function."""

    def test_empty_messages_raises_exception(self):
        """Should raise AgentNodeRoutingException for empty message list."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="No messages in state - cannot route after agent",
        ):
            validate_last_message_is_AI([])

    def test_human_message_raises_exception(self):
        """Should raise AgentNodeRoutingException when last message is HumanMessage."""
        messages: list[AnyMessage] = [HumanMessage(content="Hello")]

        with pytest.raises(
            AgentNodeRoutingException,
            match="Last message is not AIMessage \\(type: HumanMessage\\)",
        ):
            validate_last_message_is_AI(messages)

    def test_system_message_raises_exception(self):
        """Should raise AgentNodeRoutingException when last message is SystemMessage."""
        messages: list[AnyMessage] = [
            SystemMessage(content="You are a helpful assistant")
        ]

        with pytest.raises(
            AgentNodeRoutingException,
            match="Last message is not AIMessage \\(type: SystemMessage\\)",
        ):
            validate_last_message_is_AI(messages)

    def test_ai_message_returns_message(self):
        """Should return the AIMessage when it is the last message."""
        ai_message = AIMessage(content="Hello, how can I help?")
        messages: list[AnyMessage] = [ai_message]

        result = validate_last_message_is_AI(messages)

        assert result is ai_message
        assert isinstance(result, AIMessage)

    def test_ai_message_with_tool_calls_returns_message(self):
        """Should return AIMessage with tool calls when it is the last message."""
        ai_message = AIMessage(
            content="Using tool",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
        )
        messages: list[AnyMessage] = [HumanMessage(content="query"), ai_message]

        result = validate_last_message_is_AI(messages)

        assert result is ai_message
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"
        assert result.tool_calls[0]["id"] == "call_1"

    def test_mixed_messages_with_ai_last(self):
        """Should return the last AIMessage in a mixed message list."""
        ai_message = AIMessage(content="Final response")
        messages: list[AnyMessage] = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User query"),
            ai_message,
        ]

        result = validate_last_message_is_AI(messages)

        assert result is ai_message

    def test_mixed_messages_with_human_last_raises_exception(self):
        """Should raise when HumanMessage follows AIMessage."""
        messages: list[AnyMessage] = [
            AIMessage(content="AI response"),
            HumanMessage(content="Follow-up question"),
        ]

        with pytest.raises(
            AgentNodeRoutingException,
            match="Last message is not AIMessage \\(type: HumanMessage\\)",
        ):
            validate_last_message_is_AI(messages)
