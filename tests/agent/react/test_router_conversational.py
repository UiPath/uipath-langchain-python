"""Tests for router_conversational.py module."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.router_conversational import (
    create_route_agent_conversational,
)
from uipath_langchain.agent.react.types import AgentGraphNode, AgentGraphState


class MockInnerState(BaseModel):
    """Mock inner state for testing."""

    termination: Any = None
    job_attachments: dict[str, Any] = {}


class MockAgentGraphState(BaseModel):
    """Mock state compatible with AgentGraphState structure."""

    messages: list[Any] = []
    inner_state: MockInnerState = MockInnerState()


# Routing targets used across the test states (tools + control nodes).
_VALID_TARGETS: list[str] = [
    AgentGraphNode.AGENT,
    AgentGraphNode.TERMINATE,
    "search_tool",
    "calculator_tool",
    "weather_tool",
    "first_tool",
    "second_tool",
    "third_tool",
]


class TestCreateRouteAgentConversational:
    """Test cases for create_route_agent_conversational function."""

    @pytest.fixture
    def route_function(self):
        """Fixture for the conversational routing function."""
        return create_route_agent_conversational(valid_targets=_VALID_TARGETS)

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

    @pytest.fixture
    def state_with_no_ai_messages(self):
        """Fixture for state with no AI messages."""
        return MockAgentGraphState(
            messages=[
                HumanMessage(content="query"),
                HumanMessage(content="follow-up"),
            ]
        )

    def test_routes_to_single_tool_node(
        self, route_function, state_with_single_tool_call
    ):
        """Should return single tool name when AI message has one tool call."""
        result = route_function(state_with_single_tool_call)

        assert result == "search_tool"
        assert isinstance(result, str)

    def test_routes_to_first_tool_node_for_sequential_execution(
        self, route_function, state_with_multiple_tool_calls
    ):
        """Should return first tool name for sequential execution."""
        result = route_function(state_with_multiple_tool_calls)

        assert result == "search_tool"
        assert isinstance(result, str)

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
        with pytest.raises(AgentRuntimeError) as exc_info:
            route_function(empty_state)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )

    def test_no_ai_message_raises_exception(
        self, route_function, state_with_no_ai_messages
    ):
        """Should raise AgentNodeRoutingException when no AIMessage is found."""
        with pytest.raises(AgentRuntimeError) as exc_info:
            route_function(state_with_no_ai_messages)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )

    def test_human_message_after_ai_routes_to_terminate(
        self, route_function, state_with_human_last
    ):
        """Should route to TERMINATE when AI message has no tool calls (ignoring later human messages)."""
        result = route_function(state_with_human_last)
        assert result == AgentGraphNode.TERMINATE

    def test_routes_to_first_tool_in_sequence(self, route_function):
        """Should route to first tool in sequential execution."""
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

        assert result == "first_tool"

    def test_routes_to_sequence_tool_in_sequence_when_first_tool_completed(
        self, route_function
    ):
        """Should route to first tool in sequential execution."""
        ai_message = AIMessage(
            content="Using tools in order",
            tool_calls=[
                {"name": "first_tool", "args": {}, "id": "call_1"},
                {"name": "second_tool", "args": {}, "id": "call_2"},
                {"name": "third_tool", "args": {}, "id": "call_3"},
            ],
        )
        tool_message = ToolMessage(tool_call_id="call_1")
        state = MockAgentGraphState(messages=[ai_message, tool_message])

        result = route_function(state)

        assert result == "second_tool"

    def test_routes_to_agent_when_tools_calls_completed(self, route_function):
        """Should route to first tool in sequential execution."""
        ai_message = AIMessage(
            content="Using tools in order",
            tool_calls=[
                {"name": "first_tool", "args": {}, "id": "call_1"},
                {"name": "second_tool", "args": {}, "id": "call_2"},
            ],
        )
        tool_message_1 = ToolMessage(tool_call_id="call_1")
        tool_message_2 = ToolMessage(tool_call_id="call_2")
        state = MockAgentGraphState(
            messages=[ai_message, tool_message_1, tool_message_2]
        )

        result = route_function(state)

        assert result == AgentGraphNode.AGENT


class TestRouteAgentConversationalFactory:
    """Test cases for the factory function behavior."""

    def test_returns_callable(self):
        """Should return a callable routing function."""
        result = create_route_agent_conversational(valid_targets=_VALID_TARGETS)

        assert callable(result)

    def test_each_call_returns_new_function(self):
        """Should return a new function instance each time."""
        func1 = create_route_agent_conversational(valid_targets=_VALID_TARGETS)
        func2 = create_route_agent_conversational(valid_targets=_VALID_TARGETS)

        assert func1 is not func2


class TestRouteAgentConversationalTargetValidation:
    """Test guarding of router return values against valid graph targets."""

    def test_unknown_target_raises_routing_error(self):
        """Should raise ROUTING_ERROR (SYSTEM) when the routed tool is unwired."""
        route_func = create_route_agent_conversational(
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
        route_func = create_route_agent_conversational(
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
        route_func = create_route_agent_conversational()
        ai_message = AIMessage(
            content="routing",
            tool_calls=[{"name": "unwired_tool", "args": {}, "id": "call_1"}],
        )
        state = AgentGraphState(messages=[HumanMessage(content="query"), ai_message])

        assert route_func(state) == "unwired_tool"


class TestRouteAgentConversationalCustomOutput:
    """Routing for conversational agents with custom output fields.

    When the schema declares fields beyond `uipath__agent_response_messages`,
    a GENERATE_CONVERSATIONAL_OUTPUT node sits between AGENT and TERMINATE.
    """

    def test_routes_to_generate_conversational_output_when_custom_output(self):
        """No tool calls + has_custom_output=True → GENERATE_CONVERSATIONAL_OUTPUT."""
        route_func = create_route_agent_conversational(
            valid_targets=[
                AgentGraphNode.TERMINATE,
                AgentGraphNode.GENERATE_CONVERSATIONAL_OUTPUT,
            ],
            with_generate_output_node=True,
        )
        ai_message = AIMessage(content="here is my reply", tool_calls=[])
        state = AgentGraphState(messages=[HumanMessage(content="hi"), ai_message])

        assert route_func(state) == AgentGraphNode.GENERATE_CONVERSATIONAL_OUTPUT

    def test_routes_to_terminate_when_no_custom_output(self):
        """No tool calls + has_custom_output=False → TERMINATE (existing path)."""
        route_func = create_route_agent_conversational(
            valid_targets=[AgentGraphNode.TERMINATE],
            with_generate_output_node=False,
        )
        ai_message = AIMessage(content="here is my reply", tool_calls=[])
        state = AgentGraphState(messages=[HumanMessage(content="hi"), ai_message])

        assert route_func(state) == AgentGraphNode.TERMINATE

    def test_has_custom_output_does_not_affect_tool_routing(self):
        """has_custom_output=True must not change tool-routing behavior — the
        new branch only fires when the AIMessage has no tool calls."""
        route_func = create_route_agent_conversational(
            valid_targets=[
                "real_tool",
                AgentGraphNode.TERMINATE,
                AgentGraphNode.GENERATE_CONVERSATIONAL_OUTPUT,
            ],
            with_generate_output_node=True,
        )
        ai_message = AIMessage(
            content="calling tool",
            tool_calls=[{"name": "real_tool", "args": {}, "id": "call_1"}],
        )
        state = AgentGraphState(messages=[HumanMessage(content="query"), ai_message])

        assert route_func(state) == "real_tool"

    def test_has_custom_output_default_false_preserves_legacy_routing(self):
        """The new parameter defaults to False so existing callers that don't
        opt in keep routing to TERMINATE on AGENT-without-tool-calls."""
        route_func = create_route_agent_conversational(
            valid_targets=[AgentGraphNode.TERMINATE]
        )
        ai_message = AIMessage(content="reply", tool_calls=[])
        state = AgentGraphState(messages=[HumanMessage(content="hi"), ai_message])

        assert route_func(state) == AgentGraphNode.TERMINATE
