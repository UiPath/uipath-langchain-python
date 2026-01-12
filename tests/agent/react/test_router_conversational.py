"""Tests for router_conversational.py module."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from uipath_langchain.agent.exceptions import AgentNodeRoutingException
from uipath_langchain.agent.react.router_conversational import (
    create_route_agent_conversational,
)
from uipath_langchain.agent.react.types import AgentGraphNode


class MockInnerState(BaseModel):
    """Mock inner state for testing."""

    termination: Any = None
    job_attachments: dict[str, Any] = {}


class MockAgentGraphState(BaseModel):
    """Mock state compatible with AgentGraphState structure."""

    messages: list[Any] = []
    inner_state: MockInnerState = MockInnerState()


class TestCreateRouteAgentConversational:
    """Test cases for create_route_agent_conversational function."""

    @pytest.fixture
    def route_function(self):
        """Fixture for the conversational routing function."""
        return create_route_agent_conversational()

    @pytest.fixture
    def state_with_single_tool_call(self):
        """Fixture for state with a single tool call."""
        ai_message = AIMessage(
            content="Using tool",
            tool_calls=[
                {"name": "search_tool", "args": {"query": "test"}, "id": "call_1"}
            ],
        )
        return MockAgentGraphState(messages=[HumanMessage(content="query"), ai_message])

    @pytest.fixture
    def state_with_multiple_tool_calls(self):
        """Fixture for state with multiple parallel tool calls."""
        ai_message = AIMessage(
            content="Using multiple tools",
            tool_calls=[
                {"name": "search_tool", "args": {"query": "test"}, "id": "call_1"},
                {"name": "calculator_tool", "args": {"expr": "2+2"}, "id": "call_2"},
                {"name": "weather_tool", "args": {"city": "NYC"}, "id": "call_3"},
            ],
        )
        return MockAgentGraphState(messages=[HumanMessage(content="query"), ai_message])

    @pytest.fixture
    def state_with_no_tool_calls(self):
        """Fixture for state with AI message but no tool calls."""
        ai_message = AIMessage(content="I have answered your question.")
        return MockAgentGraphState(messages=[HumanMessage(content="query"), ai_message])

    @pytest.fixture
    def state_with_empty_tool_calls(self):
        """Fixture for state with empty tool_calls list."""
        ai_message = AIMessage(content="Response", tool_calls=[])
        return MockAgentGraphState(messages=[HumanMessage(content="query"), ai_message])

    @pytest.fixture
    def empty_state(self):
        """Fixture for state with no messages."""
        return MockAgentGraphState(messages=[])

    @pytest.fixture
    def state_with_human_last(self):
        """Fixture for state with HumanMessage as last message."""
        return MockAgentGraphState(
            messages=[
                AIMessage(content="response"),
                HumanMessage(content="follow-up"),
            ]
        )

    def test_routes_to_single_tool_node(
        self, route_function, state_with_single_tool_call
    ):
        """Should return list with single tool name when AI message has one tool call."""
        result = route_function(state_with_single_tool_call)

        assert result == ["search_tool"]
        assert isinstance(result, list)

    def test_routes_to_multiple_tool_nodes(
        self, route_function, state_with_multiple_tool_calls
    ):
        """Should return list of tool names for parallel execution."""
        result = route_function(state_with_multiple_tool_calls)

        assert result == ["search_tool", "calculator_tool", "weather_tool"]
        assert len(result) == 3

    def test_routes_to_terminate_when_no_tool_calls(
        self, route_function, state_with_no_tool_calls
    ):
        """Should route to TERMINATE when AI message has no tool calls."""
        result = route_function(state_with_no_tool_calls)

        assert result == AgentGraphNode.TERMINATE

    def test_routes_to_terminate_when_empty_tool_calls(
        self, route_function, state_with_empty_tool_calls
    ):
        """Should route to TERMINATE when tool_calls list is empty."""
        result = route_function(state_with_empty_tool_calls)

        assert result == AgentGraphNode.TERMINATE

    def test_empty_messages_raises_exception(self, route_function, empty_state):
        """Should raise AgentNodeRoutingException for empty messages."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="No messages in state",
        ):
            route_function(empty_state)

    def test_non_ai_last_message_raises_exception(
        self, route_function, state_with_human_last
    ):
        """Should raise AgentNodeRoutingException when last message is not AIMessage."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="Last message is not AIMessage",
        ):
            route_function(state_with_human_last)

    def test_preserves_tool_call_order(self, route_function):
        """Should preserve the order of tool calls in the result."""
        ai_message = AIMessage(
            content="Using tools in order",
            tool_calls=[
                {"name": "first_tool", "args": {}, "id": "call_1"},
                {"name": "second_tool", "args": {}, "id": "call_2"},
                {"name": "third_tool", "args": {}, "id": "call_3"},
            ],
        )
        state = MockAgentGraphState(messages=[ai_message])

        result = route_function(state)

        assert result == ["first_tool", "second_tool", "third_tool"]


class TestRouteAgentConversationalFactory:
    """Test cases for the factory function behavior."""

    def test_returns_callable(self):
        """Should return a callable routing function."""
        result = create_route_agent_conversational()

        assert callable(result)

    def test_each_call_returns_new_function(self):
        """Should return a new function instance each time."""
        func1 = create_route_agent_conversational()
        func2 = create_route_agent_conversational()

        assert func1 is not func2
