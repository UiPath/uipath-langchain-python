"""Tests for terminate_node.py module with conversational agent support."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from uipath.core.chat import UiPathConversationMessageData
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.terminate_node import create_terminate_node


class MockInnerState(BaseModel):
    """Mock inner state for testing."""

    job_attachments: dict[str, Any] = {}
    initial_message_count: int | None = None


class MockAgentGraphState(BaseModel):
    """Mock state compatible with AgentGraphState structure."""

    messages: list[Any] = []
    inner_state: MockInnerState = MockInnerState()


class TestTerminateNodeConversational:
    """Test cases for create_terminate_node with is_conversational=True."""

    def test_conversational_requires_response_schema(self):
        """Conversational mode should raise error if no response_schema provided."""

        terminate_node_no_schema = create_terminate_node(
            response_schema=None, is_conversational=True
        )
        state = MockAgentGraphState(
            messages=[
                HumanMessage(content="Initial message"),
                AIMessage(content="Response"),
            ],
            inner_state=MockInnerState(initial_message_count=1),
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            terminate_node_no_schema(state)

        assert "No response schema" in exc_info.value.error_info.title

    def test_conversational_requires_initial_message_count(self):
        """Conversational mode should raise error if initial_message_count not set."""

        class ResponseSchema(BaseModel):
            uipath__agent_response_messages: list[UiPathConversationMessageData]

        terminate_node = create_terminate_node(
            response_schema=ResponseSchema, is_conversational=True
        )
        state = MockAgentGraphState(
            messages=[AIMessage(content="Response")],
            inner_state=MockInnerState(initial_message_count=None),
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            terminate_node(state)

        assert "No initial message count" in exc_info.value.error_info.title

    def test_conversational_returns_converted_messages(self):
        """Conversational mode should return converted new messages."""

        class ResponseSchema(BaseModel):
            uipath__agent_response_messages: list[UiPathConversationMessageData]

        terminate_node = create_terminate_node(
            response_schema=ResponseSchema, is_conversational=True
        )

        # Create state with initial message count of 2, and 3 total messages
        # So only the last message should be converted
        state = MockAgentGraphState(
            messages=[
                HumanMessage(content="Initial user message"),
                AIMessage(content="Initial AI response"),
                AIMessage(content="New AI response"),
            ],
            inner_state=MockInnerState(initial_message_count=2),
        )

        result = terminate_node(state)

        assert "uipath__agent_response_messages" in result
        messages = result["uipath__agent_response_messages"]

        # Should have 1 message (only the new one after initial_message_count)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["contentParts"]) == 1
        assert messages[0]["contentParts"][0]["mimeType"] == "text/markdown"
        assert "New AI response" in str(messages[0]["contentParts"][0]["data"])

    def test_conversational_handles_multiple_new_messages(self):
        """Conversational mode should convert all messages after initial count."""

        class ResponseSchema(BaseModel):
            uipath__agent_response_messages: list[UiPathConversationMessageData]

        terminate_node = create_terminate_node(
            response_schema=ResponseSchema, is_conversational=True
        )

        # Initial count is 1, so messages at index 1+ are new
        state = MockAgentGraphState(
            messages=[
                HumanMessage(content="Initial message"),
                AIMessage(content="First new response"),
                AIMessage(content="Second new response"),
            ],
            inner_state=MockInnerState(initial_message_count=1),
        )

        result = terminate_node(state)

        messages = result["uipath__agent_response_messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert "First new response" in str(messages[0]["contentParts"][0]["data"])
        assert messages[1]["role"] == "assistant"
        assert "Second new response" in str(messages[1]["contentParts"][0]["data"])

    def test_conversational_with_tool_calls_excludes_tool_results(self):
        """Conversational mode should exclude tool results in output."""

        class ResponseSchema(BaseModel):
            uipath__agent_response_messages: list[UiPathConversationMessageData]

        terminate_node = create_terminate_node(
            response_schema=ResponseSchema, is_conversational=True
        )

        # Initial count is 1
        state = MockAgentGraphState(
            messages=[
                HumanMessage(content="Initial"),
                AIMessage(
                    content="Using tool",
                    tool_calls=[
                        {"name": "test_tool", "args": {"param": "value"}, "id": "call1"}
                    ],
                ),
                ToolMessage(content="Tool result", tool_call_id="call1"),
            ],
            inner_state=MockInnerState(initial_message_count=1),
        )

        result = terminate_node(state)

        print(result)

        messages = result["uipath__agent_response_messages"]
        # Should have AI message with tool calls, but NOT the ToolMessage
        # The mapper with include_tool_results=False should only return AI messages
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "Using tool" in str(messages[0]["contentParts"][0]["data"])
        # Verify tool calls are present in the message
        assert len(messages[0]["toolCalls"]) == 1
        assert messages[0]["toolCalls"][0]["name"] == "test_tool"
        assert messages[0]["toolCalls"][0]["input"] == {"param": "value"}

    def test_conversational_ignores_end_execution_tool(self):
        """Conversational mode should ignore END_EXECUTION tool calls."""

        class ResponseSchema(BaseModel):
            uipath__agent_response_messages: list[UiPathConversationMessageData]

        terminate_node = create_terminate_node(
            response_schema=ResponseSchema, is_conversational=True
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
        state = MockAgentGraphState(
            messages=[HumanMessage(content="Initial"), ai_message],
            inner_state=MockInnerState(initial_message_count=1),
        )

        # Should process normally, not treat as special
        result = terminate_node(state)

        assert "uipath__agent_response_messages" in result
        messages = result["uipath__agent_response_messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "Done" in str(messages[0]["contentParts"][0]["data"])


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
        with pytest.raises(AgentRuntimeError) as exc_info:
            terminate_node(state_with_raise_error)

        assert "Something went wrong" in exc_info.value.error_info.title

    def test_non_conversational_validates_ai_message(
        self, terminate_node, state_with_human_last
    ):
        """Non-conversational mode should raise if last message is not AIMessage."""
        with pytest.raises(AgentRuntimeError) as exc_info:
            terminate_node(state_with_human_last)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )

    def test_non_conversational_raises_on_no_control_flow_tool(
        self, terminate_node, state_with_no_control_flow_tool
    ):
        """Non-conversational mode should raise if no control flow tool found."""
        with pytest.raises(AgentRuntimeError) as exc_info:
            terminate_node(state_with_no_control_flow_tool)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.ROUTING_ERROR
        )


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

        with pytest.raises(AgentRuntimeError):
            terminate_node(state)
