"""Tests for agent terminate node."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from uipath_langchain.agent.react.exceptions import (
    AgentNodeRoutingException,
    AgentTerminationException,
)
from uipath_langchain.agent.react.terminate_node import create_terminate_node
from uipath_langchain.agent.react.types import AgentGraphState


class TestCreateTerminateNode:
    """Test cases for create_terminate_node function."""

    def test_returns_callable(self):
        """Should return a callable function."""
        node = create_terminate_node()
        assert callable(node)

    def test_extracts_end_execution_args(self):
        """Should extract and return args from end_execution tool call."""
        node = create_terminate_node()
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"success": True, "message": "completed"},
                            "id": "call_1",
                        }
                    ],
                ),
            ]
        )
        result = node(state)
        assert result["success"] is True
        assert result["message"] == "completed"

    def test_validates_against_custom_schema(self):
        """Should validate output against custom response schema."""

        class CustomOutput(BaseModel):
            result: str
            score: int

        node = create_terminate_node(response_schema=CustomOutput)
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"result": "done", "score": 100},
                            "id": "call_1",
                        }
                    ],
                ),
            ]
        )
        result = node(state)
        assert result["result"] == "done"
        assert result["score"] == 100

    def test_raises_on_raise_error_tool(self):
        """Should raise AgentTerminationException when raise_error tool is used."""
        node = create_terminate_node()
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "raise_error",
                            "args": {
                                "message": "Something went wrong",
                                "details": "Error details here",
                            },
                            "id": "call_1",
                        }
                    ],
                ),
            ]
        )
        with pytest.raises(AgentTerminationException) as exc_info:
            node(state)
        assert "Something went wrong" in str(exc_info.value.error_info.title)

    def test_raises_on_non_ai_last_message(self):
        """Should raise when last message is not AIMessage."""
        node = create_terminate_node()
        state = AgentGraphState(messages=[HumanMessage(content="query")])
        with pytest.raises(AgentNodeRoutingException, match="Expected last message"):
            node(state)

    def test_raises_when_no_control_flow_tool(self):
        """Should raise when no control flow tool found in terminate node."""
        node = create_terminate_node()
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "regular_tool", "args": {}, "id": "call_1"}],
                ),
            ]
        )
        with pytest.raises(AgentNodeRoutingException, match="No control flow tool"):
            node(state)

    def test_raise_error_with_default_message(self):
        """Should use default message when raise_error has no message arg."""
        node = create_terminate_node()
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "raise_error", "args": {}, "id": "call_1"}],
                ),
            ]
        )
        with pytest.raises(AgentTerminationException) as exc_info:
            node(state)
        assert "LLM did not set the error message" in str(
            exc_info.value.error_info.title
        )

    def test_processes_first_matching_control_flow_tool(self):
        """Should process the first control flow tool when multiple tool calls exist."""
        node = create_terminate_node()
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"success": True, "message": "first"},
                            "id": "call_1",
                        },
                        {
                            "name": "raise_error",
                            "args": {"message": "second"},
                            "id": "call_2",
                        },
                    ],
                ),
            ]
        )
        result = node(state)
        assert result["success"] is True
        assert result["message"] == "first"
