"""Tests for agent init node."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from uipath_langchain.agent.react.init_node import create_init_node


class TestCreateInitNode:
    """Test cases for create_init_node function."""

    def test_returns_callable(self):
        """Should return a callable function."""
        messages = [SystemMessage(content="You are a helpful assistant.")]
        node = create_init_node(messages)
        assert callable(node)

    def test_static_messages(self):
        """Should return static messages when sequence is provided."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
        ]
        node = create_init_node(messages)

        class MockState(BaseModel):
            pass

        result = node(MockState())
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], SystemMessage)
        assert isinstance(result["messages"][1], HumanMessage)

    def test_callable_messages(self):
        """Should call function to resolve messages when callable is provided."""

        def message_factory(state):
            return [
                SystemMessage(content="System prompt"),
                HumanMessage(content=f"Query: {state.query}"),
            ]

        node = create_init_node(message_factory)

        class MockState(BaseModel):
            query: str = "test question"

        result = node(MockState())
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert "Query: test question" in result["messages"][1].content

    def test_empty_messages_list(self):
        """Should handle empty messages list."""
        node = create_init_node([])

        class MockState(BaseModel):
            pass

        result = node(MockState())
        assert result["messages"] == []

    def test_preserves_message_order(self):
        """Should preserve the order of messages."""
        messages = [
            SystemMessage(content="First"),
            HumanMessage(content="Second"),
            SystemMessage(content="Third"),
        ]
        node = create_init_node(messages)

        class MockState(BaseModel):
            pass

        result = node(MockState())
        assert result["messages"][0].content == "First"
        assert result["messages"][1].content == "Second"
        assert result["messages"][2].content == "Third"

    def test_callable_receives_state(self):
        """Should pass full state to callable."""
        received_state = None

        def capture_state(state):
            nonlocal received_state
            received_state = state
            return [HumanMessage(content="test")]

        node = create_init_node(capture_state)

        class MockState(BaseModel):
            field1: str = "value1"
            field2: int = 42

        input_state = MockState()
        node(input_state)

        assert received_state is not None
        assert received_state.field1 == "value1"
        assert received_state.field2 == 42
