"""Tests for terminate_node.py module with conversational agent support."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from uipath_langchain.agent.exceptions import (
    AgentNodeRoutingException,
    AgentTerminationException,
)
from uipath_langchain.agent.react.terminate_node import create_terminate_node
from uipath_langchain.agent.react.types import (
    AgentTermination,
    AgentTerminationSource,
)


class MockInnerState(BaseModel):
    """Mock inner state for testing."""

    termination: AgentTermination | None = None
    job_attachments: dict[str, Any] = {}


class MockAgentGraphState(BaseModel):
    """Mock state compatible with AgentGraphState structure."""

    messages: list[Any] = []
    inner_state: MockInnerState = MockInnerState()


class TestTerminateNodeConversational:
    """Test cases for create_terminate_node with is_conversational=True."""

    @pytest.fixture
    def terminate_node(self):
        """Fixture for conversational terminate node."""
        return create_terminate_node(response_schema=None, is_conversational=True)

    @pytest.fixture
    def state_with_ai_message(self):
        """Fixture for state with AI message (no tool calls)."""
        return MockAgentGraphState(
            messages=[AIMessage(content="Here is my response to your question.")]
        )

    @pytest.fixture
    def state_with_human_message(self):
        """Fixture for state with human message as last."""
        return MockAgentGraphState(messages=[HumanMessage(content="User message")])

    @pytest.fixture
    def state_with_termination(self):
        """Fixture for state with termination set (e.g., escalation)."""
        termination = AgentTermination(
            source=AgentTerminationSource.ESCALATION,
            title="Escalation required",
            detail="User requested human assistance",
        )
        return MockAgentGraphState(
            messages=[AIMessage(content="response")],
            inner_state=MockInnerState(termination=termination),
        )

    def test_conversational_returns_none_no_tool_calls(
        self, terminate_node, state_with_ai_message
    ):
        """Conversational mode should return None when AI has no tool calls."""
        result = terminate_node(state_with_ai_message)

        assert result is None

    def test_conversational_skips_ai_message_validation(
        self, terminate_node, state_with_human_message
    ):
        """Conversational mode should not validate that last message is AIMessage."""
        # This should not raise, unlike non-conversational mode
        result = terminate_node(state_with_human_message)

        assert result is None

    def test_conversational_handles_termination(
        self, terminate_node, state_with_termination
    ):
        """Conversational mode should still handle state.inner_state.termination."""
        with pytest.raises(AgentTerminationException) as exc_info:
            terminate_node(state_with_termination)

        assert "Escalation required" in exc_info.value.error_info.title

    def test_conversational_ignores_end_execution_tool(self):
        """Conversational mode should ignore END_EXECUTION tool calls."""
        terminate_node = create_terminate_node(
            response_schema=None, is_conversational=True
        )
        ai_message = AIMessage(
            content="Done",
            tool_calls=[
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {"result": "completed"},
                    "id": "call_1",
                }
            ],
        )
        state = MockAgentGraphState(messages=[ai_message])

        # Should return None, not process the tool call
        result = terminate_node(state)

        assert result is None


class TestTerminateNodeNonConversational:
    """Test cases for create_terminate_node with is_conversational=False (default)."""

    @pytest.fixture
    def terminate_node(self):
        """Fixture for non-conversational terminate node."""
        return create_terminate_node(response_schema=None, is_conversational=False)

    @pytest.fixture
    def state_with_end_execution(self):
        """Fixture for state with END_EXECUTION tool call."""
        ai_message = AIMessage(
            content="Task completed",
            tool_calls=[
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {"success": True, "message": "Task completed successfully"},
                    "id": "call_1",
                }
            ],
        )
        return MockAgentGraphState(messages=[ai_message])

    @pytest.fixture
    def state_with_raise_error(self):
        """Fixture for state with RAISE_ERROR tool call."""
        ai_message = AIMessage(
            content="Error occurred",
            tool_calls=[
                {
                    "name": RAISE_ERROR_TOOL.name,
                    "args": {
                        "message": "Something went wrong",
                        "details": "Additional info",
                    },
                    "id": "call_1",
                }
            ],
        )
        return MockAgentGraphState(messages=[ai_message])

    @pytest.fixture
    def state_with_human_last(self):
        """Fixture for state with HumanMessage as last message."""
        return MockAgentGraphState(messages=[HumanMessage(content="User message")])

    @pytest.fixture
    def state_with_no_control_flow_tool(self):
        """Fixture for state with AI message but no control flow tool."""
        ai_message = AIMessage(
            content="Using regular tool",
            tool_calls=[{"name": "regular_tool", "args": {}, "id": "call_1"}],
        )
        return MockAgentGraphState(messages=[ai_message])

    @pytest.fixture
    def state_with_termination(self):
        """Fixture for state with escalation termination."""
        termination = AgentTermination(
            source=AgentTerminationSource.ESCALATION,
            title="Needs human review",
            detail="Complex issue",
        )
        return MockAgentGraphState(
            messages=[AIMessage(content="response")],
            inner_state=MockInnerState(termination=termination),
        )

    def test_non_conversational_handles_end_execution(
        self, terminate_node, state_with_end_execution
    ):
        """Non-conversational mode should process END_EXECUTION tool and return validated output."""
        result = terminate_node(state_with_end_execution)

        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert result["message"] == "Task completed successfully"

    def test_non_conversational_handles_raise_error(
        self, terminate_node, state_with_raise_error
    ):
        """Non-conversational mode should process RAISE_ERROR tool and raise exception."""
        with pytest.raises(AgentTerminationException) as exc_info:
            terminate_node(state_with_raise_error)

        assert "Something went wrong" in exc_info.value.error_info.title

    def test_non_conversational_validates_ai_message(
        self, terminate_node, state_with_human_last
    ):
        """Non-conversational mode should raise if last message is not AIMessage."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="Expected last message to be AIMessage, got HumanMessage",
        ):
            terminate_node(state_with_human_last)

    def test_non_conversational_raises_on_no_control_flow_tool(
        self, terminate_node, state_with_no_control_flow_tool
    ):
        """Non-conversational mode should raise if no control flow tool found."""
        with pytest.raises(
            AgentNodeRoutingException,
            match="No control flow tool call found in terminate node",
        ):
            terminate_node(state_with_no_control_flow_tool)

    def test_non_conversational_handles_termination_first(
        self, terminate_node, state_with_termination
    ):
        """Non-conversational mode should check termination before tool calls."""
        with pytest.raises(AgentTerminationException) as exc_info:
            terminate_node(state_with_termination)

        assert "Needs human review" in exc_info.value.error_info.title


class TestTerminateNodeWithResponseSchema:
    """Test cases for terminate node with custom response schema."""

    def test_end_execution_with_custom_schema(self):
        """Should validate output against custom response schema."""

        class CustomOutput(BaseModel):
            status: str
            count: int

        terminate_node = create_terminate_node(
            response_schema=CustomOutput, is_conversational=False
        )
        ai_message = AIMessage(
            content="Done",
            tool_calls=[
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {"status": "completed", "count": 42},
                    "id": "call_1",
                }
            ],
        )
        state = MockAgentGraphState(messages=[ai_message])

        result = terminate_node(state)

        assert result == {"status": "completed", "count": 42}


class TestTerminateNodeFactory:
    """Test cases for the factory function behavior."""

    def test_returns_callable(self):
        """Should return a callable terminate node function."""
        result = create_terminate_node(response_schema=None, is_conversational=False)

        assert callable(result)

    def test_default_is_non_conversational(self):
        """Default should be non-conversational mode."""
        # Create without is_conversational param
        terminate_node = create_terminate_node(response_schema=None)

        # Should behave as non-conversational (raise on non-AI last message)
        state = MockAgentGraphState(messages=[HumanMessage(content="test")])

        with pytest.raises(AgentNodeRoutingException):
            terminate_node(state)
