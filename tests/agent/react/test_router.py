"""Tests for router.py module."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL

from uipath_langchain.agent.exceptions import AgentNodeRoutingException
from uipath_langchain.agent.react.router import create_route_agent
from uipath_langchain.agent.react.types import AgentGraphNode


class MockInnerState(BaseModel):
    """Mock inner state for testing."""

    termination: None = None
    job_attachments: dict[str, Any] = {}


class MockAgentGraphState(BaseModel):
    """Mock state compatible with AgentGraphState structure."""

    messages: list[AnyMessage] = []
    inner_state: MockInnerState = MockInnerState()


# Module-level fixtures available to all test classes


@pytest.fixture
def route_function_no_limit():
    """Fixture for routing function with no thinking messages limit."""
    return create_route_agent(thinking_messages_limit=0)


@pytest.fixture
def route_function_with_limit():
    """Fixture for routing function with thinking messages limit of 2."""
    return create_route_agent(thinking_messages_limit=2)


@pytest.fixture
def state_single_tool_call():
    """Fixture for state with a single tool call."""
    ai_message = AIMessage(
        content="Using search tool",
        tool_calls=[{"name": "search_tool", "args": {"query": "test"}, "id": "call_1"}],
    )
    return MockAgentGraphState(messages=[HumanMessage(content="query"), ai_message])


@pytest.fixture
def state_multiple_tool_calls():
    """Fixture for state with multiple tool calls (sequential execution)."""
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
def state_partial_execution():
    """Fixture for state with partially executed tool calls."""
    ai_message = AIMessage(
        content="Using multiple tools",
        tool_calls=[
            {"name": "search_tool", "args": {"query": "test"}, "id": "call_1"},
            {"name": "calculator_tool", "args": {"expr": "2+2"}, "id": "call_2"},
            {"name": "weather_tool", "args": {"city": "NYC"}, "id": "call_3"},
        ],
    )
    tool_message = ToolMessage(content="search result", tool_call_id="call_1")
    return MockAgentGraphState(
        messages=[HumanMessage(content="query"), ai_message, tool_message]
    )


@pytest.fixture
def state_all_tools_executed():
    """Fixture for state with all tool calls executed."""
    ai_message = AIMessage(
        content="Using two tools",
        tool_calls=[
            {"name": "search_tool", "args": {"query": "test"}, "id": "call_1"},
            {"name": "calculator_tool", "args": {"expr": "2+2"}, "id": "call_2"},
        ],
    )
    tool_message_1 = ToolMessage(content="search result", tool_call_id="call_1")
    tool_message_2 = ToolMessage(content="calc result", tool_call_id="call_2")
    return MockAgentGraphState(
        messages=[
            HumanMessage(content="query"),
            ai_message,
            tool_message_1,
            tool_message_2,
        ]
    )


@pytest.fixture
def state_flow_control_tool():
    """Fixture for state with flow control tool call."""
    ai_message = AIMessage(
        content="Ending execution",
        tool_calls=[
            {
                "name": END_EXECUTION_TOOL.name,
                "args": {"reason": "done"},
                "id": "call_1",
            }
        ],
    )
    return MockAgentGraphState(messages=[HumanMessage(content="query"), ai_message])


@pytest.fixture
def state_no_tool_calls():
    """Fixture for state with AI message but no tool calls."""
    ai_message = AIMessage(content="I have answered your question.")
    return MockAgentGraphState(messages=[HumanMessage(content="query"), ai_message])


@pytest.fixture
def state_excessive_thinking():
    """Fixture for state with excessive consecutive thinking messages."""
    messages = [
        HumanMessage(content="query"),
        AIMessage(content="thinking 1"),
        AIMessage(content="thinking 2"),
        AIMessage(content="thinking 3"),
    ]
    return MockAgentGraphState(messages=messages)


@pytest.fixture
def empty_state():
    """Fixture for state with no messages."""
    return MockAgentGraphState(messages=[])


@pytest.fixture
def state_no_ai_messages():
    """Fixture for state with no AI messages."""
    return MockAgentGraphState(messages=[HumanMessage(content="test")])


class TestRouteAgentBasicFunctionality:
    """Test basic routing functionality."""

    def test_single_tool_call_sequential_execution(
        self, route_function_no_limit, state_single_tool_call
    ):
        """Should return tool name for single tool call."""
        result = route_function_no_limit(state_single_tool_call)
        assert result == "search_tool"
        assert isinstance(result, str)

    def test_multiple_tool_calls_first_tool(
        self, route_function_no_limit, state_multiple_tool_calls
    ):
        """Should return first tool name for sequential execution."""
        result = route_function_no_limit(state_multiple_tool_calls)
        assert result == "search_tool"

    def test_partial_execution_next_tool(
        self, route_function_no_limit, state_partial_execution
    ):
        """Should return next unexecuted tool name."""
        result = route_function_no_limit(state_partial_execution)
        assert result == "calculator_tool"

    def test_all_tools_executed_back_to_agent(
        self, route_function_no_limit, state_all_tools_executed
    ):
        """Should route back to AGENT when all tools are executed."""
        result = route_function_no_limit(state_all_tools_executed)
        assert result == AgentGraphNode.AGENT

    def test_flow_control_tool_terminates(
        self, route_function_no_limit, state_flow_control_tool
    ):
        """Should route to TERMINATE for flow control tools."""
        result = route_function_no_limit(state_flow_control_tool)
        assert result == AgentGraphNode.TERMINATE


class TestRouteAgentThinkingMessages:
    """Test thinking messages and consecutive completions logic."""

    def test_no_tool_calls_within_limit_routes_to_agent(
        self, route_function_with_limit, state_no_tool_calls
    ):
        """Should route to AGENT when no tool calls and within thinking limit."""
        result = route_function_with_limit(state_no_tool_calls)
        assert result == AgentGraphNode.AGENT

    def test_excessive_thinking_messages_raises_exception(
        self, route_function_with_limit, state_excessive_thinking
    ):
        """Should raise exception when exceeding thinking messages limit."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="Agent exceeded consecutive completions limit",
        ):
            route_function_with_limit(state_excessive_thinking)

    def test_thinking_messages_limit_zero_forbids_thinking(self):
        """Should not allow any thinking messages when limit is 0."""
        route_func = create_route_agent(thinking_messages_limit=0)
        ai_message = AIMessage(content="thinking")
        state = MockAgentGraphState(
            messages=[HumanMessage(content="query"), ai_message]
        )

        with pytest.raises(
            AgentNodeRoutingException,
            match="Agent exceeded consecutive completions limit",
        ):
            route_func(state)

    def test_thinking_messages_after_tool_execution_resets_count(self):
        """Should reset thinking count after tool execution."""
        route_func = create_route_agent(thinking_messages_limit=1)
        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="using tool",
                tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(content="result", tool_call_id="call_1"),
            AIMessage(content="thinking after tool"),  # This should be allowed
        ]
        state = MockAgentGraphState(messages=messages)

        result = route_func(state)
        assert result == AgentGraphNode.AGENT


class TestRouteAgentErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_messages_raises_exception(
        self, route_function_no_limit, empty_state
    ):
        """Should raise exception for empty messages."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="No AIMessage found in messages for routing",
        ):
            route_function_no_limit(empty_state)

    def test_no_ai_messages_raises_exception(
        self, route_function_no_limit, state_no_ai_messages
    ):
        """Should raise exception when no AI messages found."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="No AIMessage found in messages for routing",
        ):
            route_function_no_limit(state_no_ai_messages)

    def test_empty_ai_response_raises_exception(self, route_function_no_limit):
        """Should raise exception for empty AI response without tool calls."""
        ai_message = AIMessage(content="")  # Empty content
        state = MockAgentGraphState(
            messages=[HumanMessage(content="query"), ai_message]
        )

        with pytest.raises(
            AgentNodeRoutingException,
            match="Agent produced empty response without tool calls",
        ):
            route_function_no_limit(state)


class TestRouteAgentMultipleToolCallSequencing:
    """Test sequential dispatching of multiple tool calls in a single AI message."""

    def test_three_tools_dispatched_sequentially(self):
        """Router dispatches each tool in order when multiple tool calls exist."""
        route_func = create_route_agent(thinking_messages_limit=0)

        ai_message = AIMessage(
            content="Using three tools",
            tool_calls=[
                {"name": "tool_a", "args": {}, "id": "call_a"},
                {"name": "tool_b", "args": {}, "id": "call_b"},
                {"name": "tool_c", "args": {}, "id": "call_c"},
            ],
        )

        # Step 1: No tool results yet — route to first tool
        state_0 = MockAgentGraphState(
            messages=[HumanMessage(content="query"), ai_message]
        )
        assert route_func(state_0) == "tool_a"

        # Step 2: First tool done — route to second
        state_1 = MockAgentGraphState(
            messages=[
                HumanMessage(content="query"),
                ai_message,
                ToolMessage(content="result_a", tool_call_id="call_a"),
            ]
        )
        assert route_func(state_1) == "tool_b"

        # Step 3: Two tools done — route to third
        state_2 = MockAgentGraphState(
            messages=[
                HumanMessage(content="query"),
                ai_message,
                ToolMessage(content="result_a", tool_call_id="call_a"),
                ToolMessage(content="result_b", tool_call_id="call_b"),
            ]
        )
        assert route_func(state_2) == "tool_c"

        # Step 4: All done — route back to agent
        state_3 = MockAgentGraphState(
            messages=[
                HumanMessage(content="query"),
                ai_message,
                ToolMessage(content="result_a", tool_call_id="call_a"),
                ToolMessage(content="result_b", tool_call_id="call_b"),
                ToolMessage(content="result_c", tool_call_id="call_c"),
            ]
        )
        assert route_func(state_3) == AgentGraphNode.AGENT

    def test_flow_control_tool_among_multiple_terminates(self):
        """Router should terminate when the next tool is a flow control tool."""
        route_func = create_route_agent(thinking_messages_limit=0)

        ai_message = AIMessage(
            content="Using tools then ending",
            tool_calls=[
                {"name": "regular_tool", "args": {}, "id": "call_1"},
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {"reason": "done"},
                    "id": "call_2",
                },
            ],
        )

        # First tool is regular
        state_0 = MockAgentGraphState(
            messages=[HumanMessage(content="query"), ai_message]
        )
        assert route_func(state_0) == "regular_tool"

        # After first tool done, next is flow control — terminate
        state_1 = MockAgentGraphState(
            messages=[
                HumanMessage(content="query"),
                ai_message,
                ToolMessage(content="done", tool_call_id="call_1"),
            ]
        )
        assert route_func(state_1) == AgentGraphNode.TERMINATE
