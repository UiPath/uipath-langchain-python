"""Shared fixtures for agent tests."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from uipath_langchain.agent.react.types import AgentGraphState


@pytest.fixture
def sample_human_message():
    """Simple human message for testing."""
    return HumanMessage(content="test query")


@pytest.fixture
def sample_system_message():
    """Simple system message for testing."""
    return SystemMessage(content="You are a helpful assistant.")


@pytest.fixture
def sample_ai_message():
    """AIMessage with content, no tool calls."""
    return AIMessage(content="test response")


@pytest.fixture
def sample_ai_message_empty():
    """AIMessage with empty content, no tool calls."""
    return AIMessage(content="")


@pytest.fixture
def sample_ai_message_with_tool_call():
    """AIMessage with a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "test_tool", "args": {"param": "value"}, "id": "call_1"}],
    )


@pytest.fixture
def sample_ai_message_with_end_execution():
    """AIMessage with end_execution tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "end_execution",
                "args": {"final_answer": "completed"},
                "id": "call_end",
            }
        ],
    )


@pytest.fixture
def sample_ai_message_with_raise_error():
    """AIMessage with raise_error tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "raise_error",
                "args": {"message": "error occurred", "details": "some details"},
                "id": "call_error",
            }
        ],
    )


@pytest.fixture
def sample_tool_message():
    """Tool message response."""
    return ToolMessage(content="tool result", tool_call_id="call_1")


@pytest.fixture
def sample_agent_state_empty():
    """AgentGraphState with no messages."""
    return AgentGraphState(messages=[])


@pytest.fixture
def sample_agent_state_with_human_message(sample_human_message):
    """AgentGraphState with single human message."""
    return AgentGraphState(messages=[sample_human_message])


@pytest.fixture
def sample_agent_state_with_ai_response(sample_human_message, sample_ai_message):
    """AgentGraphState with human query and AI response."""
    return AgentGraphState(messages=[sample_human_message, sample_ai_message])


@pytest.fixture
def sample_agent_state_with_tool_call(
    sample_human_message, sample_ai_message_with_tool_call
):
    """AgentGraphState with human query and AI tool call."""
    return AgentGraphState(
        messages=[sample_human_message, sample_ai_message_with_tool_call]
    )


@pytest.fixture
def sample_agent_state_with_end_execution(
    sample_human_message, sample_ai_message_with_end_execution
):
    """AgentGraphState with end_execution tool call."""
    return AgentGraphState(
        messages=[sample_human_message, sample_ai_message_with_end_execution]
    )
