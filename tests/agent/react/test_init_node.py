"""Tests for init_node.py module with conversational agent support."""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite
from pydantic import BaseModel

from uipath_langchain.agent.react.init_node import create_init_node


class MockState(BaseModel):
    """Mock state for testing init node."""

    messages: list[Any] = []


class TestCreateInitNodeConversational:
    """Test cases for create_init_node with is_conversational=True."""

    @pytest.fixture
    def system_message(self):
        """Fixture for a system message."""
        return SystemMessage(
            content="You are a helpful assistant. Current date: 2024-01-15"
        )

    @pytest.fixture
    def new_system_message(self):
        """Fixture for a new system message (simulating resume with new date)."""
        return SystemMessage(
            content="You are a helpful assistant. Current date: 2024-01-16"
        )

    @pytest.fixture
    def user_message(self):
        """Fixture for a user message."""
        return HumanMessage(content="Hello, how are you?")

    @pytest.fixture
    def empty_state(self):
        """Fixture for state with no messages."""
        return MockState(messages=[])

    def test_conversational_empty_state_returns_overwrite(
        self, system_message, user_message
    ):
        """Conversational mode with empty state should use Overwrite with new messages."""
        messages = [system_message, user_message]
        init_node = create_init_node(
            messages, input_schema=None, is_conversational=True
        )
        state = MockState(messages=[])

        result = init_node(state)

        assert "messages" in result
        assert isinstance(result["messages"], Overwrite)
        # The Overwrite wraps the message list
        overwrite_value = result["messages"].value
        assert len(overwrite_value) == 2
        assert overwrite_value[0] == system_message
        assert overwrite_value[1] == user_message

    def test_conversational_resume_replaces_system_message(
        self, new_system_message, user_message
    ):
        """Conversational mode should replace old SystemMessage when resuming."""
        old_system_message = SystemMessage(content="Old system prompt")
        old_human_message = HumanMessage(content="Previous user message")

        # State has existing conversation with system message at position 0
        state = MockState(messages=[old_system_message, old_human_message])

        # New messages to be applied (new system message with updated date)
        new_messages = [new_system_message, user_message]
        init_node = create_init_node(
            new_messages, input_schema=None, is_conversational=True
        )

        result = init_node(state)

        assert isinstance(result["messages"], Overwrite)
        overwrite_value = result["messages"].value
        # Should have: new_system_message, user_message, old_human_message (old system message removed)
        assert len(overwrite_value) == 3
        assert overwrite_value[0] == new_system_message
        assert overwrite_value[1] == user_message
        assert overwrite_value[2] == old_human_message

    def test_conversational_resume_preserves_non_system_first_message(
        self, system_message, user_message
    ):
        """Conversational mode should preserve all messages if first is not SystemMessage."""
        # State starts with HumanMessage (not SystemMessage)
        existing_human_message = HumanMessage(content="Previous query")
        state = MockState(messages=[existing_human_message])

        new_messages = [system_message, user_message]
        init_node = create_init_node(
            new_messages, input_schema=None, is_conversational=True
        )

        result = init_node(state)

        assert isinstance(result["messages"], Overwrite)
        overwrite_value = result["messages"].value
        # Should have: system_message, user_message, existing_human_message
        assert len(overwrite_value) == 3
        assert overwrite_value[0] == system_message
        assert overwrite_value[1] == user_message
        assert overwrite_value[2] == existing_human_message

    def test_conversational_with_callable_messages(self):
        """Conversational mode should work with callable message generators."""

        def message_generator(state):
            return [
                SystemMessage(
                    content=f"System for state with {len(state.messages)} messages"
                ),
                HumanMessage(content="New query"),
            ]

        state = MockState(messages=[])
        init_node = create_init_node(
            message_generator, input_schema=None, is_conversational=True
        )

        result = init_node(state)

        assert isinstance(result["messages"], Overwrite)
        overwrite_value = result["messages"].value
        assert len(overwrite_value) == 2
        assert "System for state with 0 messages" in overwrite_value[0].content


class TestCreateInitNodeNonConversational:
    """Test cases for create_init_node with is_conversational=False (default)."""

    @pytest.fixture
    def system_message(self):
        """Fixture for a system message."""
        return SystemMessage(content="You are a helpful assistant.")

    @pytest.fixture
    def user_message(self):
        """Fixture for a user message."""
        return HumanMessage(content="Hello")

    def test_non_conversational_returns_list_not_overwrite(
        self, system_message, user_message
    ):
        """Non-conversational mode should return list, not Overwrite."""
        messages = [system_message, user_message]
        init_node = create_init_node(
            messages, input_schema=None, is_conversational=False
        )
        state = MockState(messages=[])

        result = init_node(state)

        assert "messages" in result
        # Non-conversational mode returns a list (for add_messages reducer to append)
        assert isinstance(result["messages"], list)
        assert not isinstance(result["messages"], Overwrite)
        assert len(result["messages"]) == 2

    def test_non_conversational_default_behavior(self, system_message, user_message):
        """Default behavior (no is_conversational param) should be non-conversational."""
        messages = [system_message, user_message]
        init_node = create_init_node(messages, input_schema=None)
        state = MockState(messages=[])

        result = init_node(state)

        assert isinstance(result["messages"], list)
        assert not isinstance(result["messages"], Overwrite)


class TestCreateInitNodeInnerState:
    """Test cases for init node inner_state initialization."""

    def test_returns_inner_state_with_job_attachments(self):
        """Init node should return inner_state with job_attachments dict."""
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content="System"),
            HumanMessage(content="Query"),
        ]
        init_node = create_init_node(
            messages, input_schema=None, is_conversational=False
        )
        state = MockState(messages=[])

        result = init_node(state)

        assert "inner_state" in result
        assert "job_attachments" in result["inner_state"]
        assert isinstance(result["inner_state"]["job_attachments"], dict)

    def test_inner_state_present_in_conversational_mode(self):
        """Inner state should also be present in conversational mode."""
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content="System"),
            HumanMessage(content="Query"),
        ]
        init_node = create_init_node(
            messages, input_schema=None, is_conversational=True
        )
        state = MockState(messages=[])

        result = init_node(state)

        assert "inner_state" in result
        assert "job_attachments" in result["inner_state"]
