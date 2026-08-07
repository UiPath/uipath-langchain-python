"""Tests for router.py module."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.router import create_route_agent
from uipath_langchain.agent.react.types import AgentGraphNode, AgentGraphState


class MockInnerState(BaseModel):
    """Mock inner state for testing."""

    termination: None = None
    job_attachments: dict[str, Any] = {}


class MockAgentGraphState(BaseModel):
    """Mock state compatible with AgentGraphState structure."""

    messages: list[AnyMessage] = []
    inner_state: MockInnerState = MockInnerState()


# Module-level fixtures available to all test classes

# Routing targets used across the test states (tools + control nodes).
_VALID_TARGETS: list[str] = [
    AgentGraphNode.AGENT,
    AgentGraphNode.TERMINATE,
    "search_tool",
    "calculator_tool",
    "weather_tool",
    "test_tool",
]


@pytest.fixture
def route_function_no_limit():
    """Routing function. Tool-less turns loop back; llm_messages_limit bounds them."""
    return create_route_agent(valid_targets=_VALID_TARGETS)


@pytest.fixture
def route_function_with_limit():
    """Alias kept for existing tests; routing no longer takes a thinking limit."""
    return create_route_agent(valid_targets=_VALID_TARGETS)


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
    """A tool-less turn loops back only when it carries a reasoning block (an
    expected thinking stall, which the LLM node's extraction terminates next turn).
    A tool-less turn with plain content and no reasoning means the model was forced
    but didn't call a tool, which fails fast."""

    def test_reasoning_stall_routes_to_agent(self, route_function_no_limit):
        """A tool-less turn carrying a thinking block loops back to AGENT."""
        ai_message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "the answer is 42"},
            ]
        )
        state = MockAgentGraphState(
            messages=[HumanMessage(content="query"), ai_message]
        )
        assert route_function_no_limit(state) == AgentGraphNode.AGENT

    def test_forced_tool_less_without_reasoning_raises(
        self, route_function_no_limit, state_no_tool_calls
    ):
        """Content but no tool call and no reasoning block => the model ignored forced
        tool_choice => THINKING_LIMIT_EXCEEDED, rather than looping to max iterations."""
        with pytest.raises(AgentRuntimeError) as exc_info:
            route_function_no_limit(state_no_tool_calls)
        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED
        )


class TestRouteAgentErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_messages_raises_exception(
        self, route_function_no_limit, empty_state
    ):
        """Should raise exception for empty messages."""
        with pytest.raises(AgentRuntimeError) as exc_info:
            route_function_no_limit(empty_state)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )

    def test_no_ai_messages_raises_exception(
        self, route_function_no_limit, state_no_ai_messages
    ):
        """Should raise exception when no AI messages found."""
        with pytest.raises(AgentRuntimeError) as exc_info:
            route_function_no_limit(state_no_ai_messages)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )

    def test_empty_ai_response_raises_exception(self, route_function_no_limit):
        """Should raise exception for empty AI response without tool calls."""
        ai_message = AIMessage(content="")  # Empty content
        state = MockAgentGraphState(
            messages=[HumanMessage(content="query"), ai_message]
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            route_function_no_limit(state)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )


class TestRouteAgentTargetValidation:
    """Test guarding of router return values against valid graph targets."""

    def test_unknown_target_raises_routing_error(self):
        """Should raise ROUTING_ERROR (SYSTEM) when the routed tool is unwired."""
        route_func = create_route_agent(
            valid_targets=[AgentGraphNode.AGENT, AgentGraphNode.TERMINATE, "real_tool"],
        )
        ai_message = AIMessage(
            content="routing",
            tool_calls=[{"name": "context", "args": {}, "id": "call_1"}],
        )
        state = AgentGraphState(messages=[HumanMessage(content="query"), ai_message])

        with pytest.raises(AgentRuntimeError) as exc_info:
            route_func(state)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM

    def test_known_target_returns_tool_name(self):
        """Should return the tool name when it is in the valid target set."""
        route_func = create_route_agent(
            valid_targets=[AgentGraphNode.AGENT, AgentGraphNode.TERMINATE, "real_tool"],
        )
        ai_message = AIMessage(
            content="routing",
            tool_calls=[{"name": "real_tool", "args": {}, "id": "call_1"}],
        )
        state = AgentGraphState(messages=[HumanMessage(content="query"), ai_message])

        assert route_func(state) == "real_tool"

    def test_default_valid_targets_skips_guard(self):
        """Should skip the destination guard when valid_targets is left unset.

        Backwards-compatible contract: callers predating valid_targets must keep
        the old unguarded behavior, returning any routed tool name as-is.
        """
        route_func = create_route_agent()
        ai_message = AIMessage(
            content="routing",
            tool_calls=[{"name": "unwired_tool", "args": {}, "id": "call_1"}],
        )
        state = AgentGraphState(messages=[HumanMessage(content="query"), ai_message])

        assert route_func(state) == "unwired_tool"
