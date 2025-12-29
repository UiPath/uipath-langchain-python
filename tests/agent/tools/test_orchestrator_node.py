"""Tests for orchestrator_node.py module."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from uipath_langchain.agent.exceptions import AgentNodeRoutingException
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.orchestrator_node import create_orchestrator_node


class TestOrchestratorNode:
    """Test cases for orchestrator node."""

    def test_no_messages_throws_exception(self):
        """Test that empty messages throw exception."""
        orchestrator = create_orchestrator_node()
        state = AgentGraphState(messages=[])

        with pytest.raises(AgentNodeRoutingException):
            orchestrator(state)

    def test_no_ai_message_returns_none(self):
        """Test that no AI message returns current_tool_call_index None to route back to LLM."""
        orchestrator = create_orchestrator_node()
        human_message = HumanMessage(content="Hello")
        state = AgentGraphState(messages=[human_message], current_tool_call_index=None)

        result = orchestrator(state)

        assert result == {"current_tool_call_index": None}

    def test_new_tool_call_batch_sets_index_to_zero(self):
        """Test that new tool call batch sets current_tool_call_index to 0."""
        orchestrator = create_orchestrator_node()
        tool_call = {
            "name": "test_tool",
            "args": {"input": "test"},
            "id": "call_1",
        }
        ai_message = AIMessage(content="Using tool", tool_calls=[tool_call])
        state = AgentGraphState(messages=[ai_message], current_tool_call_index=None)

        result = orchestrator(state)

        assert result == {"current_tool_call_index": 0}

    def test_thinking_messages_limit_enforcement(self):
        """Test that thinking messages limit is enforced."""
        orchestrator = create_orchestrator_node(thinking_messages_limit=2)

        # Create multiple AI messages without tool calls (thinking messages)
        messages = [
            AIMessage(content="Thinking 1"),
            AIMessage(content="Thinking 2"),
            AIMessage(content="Still thinking 3"),
        ]
        state = AgentGraphState(messages=messages, current_tool_call_index=None)

        with pytest.raises(AgentNodeRoutingException):
            orchestrator(state)

    def test_thinking_messages_limit_zero_throws_immediately(self):
        """Test that thinking_messages_limit=0 throws on any AI message without tool calls."""
        orchestrator = create_orchestrator_node(thinking_messages_limit=0)

        # Single AI message without tool calls should throw
        messages = [AIMessage(content="Just thinking")]
        state = AgentGraphState(messages=messages, current_tool_call_index=None)

        with pytest.raises(AgentNodeRoutingException):
            orchestrator(state)

    def test_flow_control_tool_filtering_single_tool(self):
        """Test that single flow control tool is not filtered."""
        orchestrator = create_orchestrator_node()
        tool_call = {
            "name": "end_execution",
            "args": {"result": "done"},
            "id": "call_1",
        }
        ai_message = AIMessage(content="Ending", tool_calls=[tool_call])
        state = AgentGraphState(messages=[ai_message], current_tool_call_index=None)

        result = orchestrator(state)

        assert result == {"current_tool_call_index": 0}

    def test_flow_control_tool_filtering_multiple_tools(self):
        """Test that flow control tools are filtered when multiple tools exist."""
        orchestrator = create_orchestrator_node()
        tool_calls = [
            {
                "name": "regular_tool",
                "args": {"input": "test"},
                "id": "call_1",
            },
            {
                "name": "end_execution",
                "args": {"result": "done"},
                "id": "call_2",
            },
        ]
        ai_message = AIMessage(content="Using tools", tool_calls=tool_calls)
        state = AgentGraphState(messages=[ai_message], current_tool_call_index=None)

        result = orchestrator(state)

        assert "messages" in result
        assert result["current_tool_call_index"] == 0

        modified_message = result["messages"][0]
        assert len(modified_message.tool_calls) == 1
        assert modified_message.tool_calls[0]["name"] == "regular_tool"

    def test_content_filtering_with_tool_calls(self):
        """Test that content blocks are filtered along with tool calls."""
        orchestrator = create_orchestrator_node()
        tool_calls = [
            {
                "name": "regular_tool",
                "args": {"input": "test"},
                "id": "call_1",
            },
            {
                "name": "end_execution",
                "args": {"result": "done"},
                "id": "call_2",
            },
        ]
        content_blocks: list[str | dict[Any, Any]] = [
            {"type": "text", "text": "Using tools"},
            {"type": "tool_use", "call_id": "call_1", "name": "regular_tool"},
            {"type": "tool_use", "call_id": "call_2", "name": "end_execution"},
        ]
        ai_message = AIMessage(content=content_blocks, tool_calls=tool_calls)
        state = AgentGraphState(messages=[ai_message], current_tool_call_index=None)

        print(ai_message.content)
        result = orchestrator(state)

        assert "messages" in result
        modified_message = result["messages"][0]
        assert len(modified_message.tool_calls) == 1
        assert modified_message.tool_calls[0]["name"] == "regular_tool"

        # Check content is filtered
        filtered_content = modified_message.content
        print(filtered_content)
        assert len(filtered_content) == 2  # text block + regular_tool block
        tool_use_blocks = [
            block
            for block in filtered_content
            if isinstance(block, dict) and block.get("call_id") is not None
        ]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["call_id"] == "call_1"

    def test_processing_batch_advancement(self):
        """Test advancement through tool call batch."""
        orchestrator = create_orchestrator_node()
        tool_calls = [
            {
                "name": "tool_1",
                "args": {"input": "test1"},
                "id": "call_1",
            },
            {
                "name": "tool_2",
                "args": {"input": "test2"},
                "id": "call_2",
            },
        ]
        ai_message = AIMessage(content="Using tools", tool_calls=tool_calls)
        state = AgentGraphState(messages=[ai_message], current_tool_call_index=0)

        result = orchestrator(state)

        assert result == {"current_tool_call_index": 1}

    def test_processing_batch_completion(self):
        """Test completion of tool call batch."""
        orchestrator = create_orchestrator_node()
        tool_calls = [
            {
                "name": "tool_1",
                "args": {"input": "test1"},
                "id": "call_1",
            },
            {
                "name": "tool_2",
                "args": {"input": "test2"},
                "id": "call_2",
            },
        ]
        ai_message = AIMessage(content="Using tools", tool_calls=tool_calls)
        state = AgentGraphState(messages=[ai_message], current_tool_call_index=1)

        result = orchestrator(state)

        assert result == {"current_tool_call_index": None}

    def test_no_latest_ai_message_in_batch_throws_exception(self):
        """Test that no latest AI message during batch processing throws exception."""
        orchestrator = create_orchestrator_node()
        human_message = HumanMessage(content="Hello")
        state = AgentGraphState(messages=[human_message], current_tool_call_index=0)

        with pytest.raises(AgentNodeRoutingException):
            orchestrator(state)

    def test_latest_ai_message_with_tool_responses_mixed(self):
        """Test finding latest AI message when mixed with tool responses."""
        orchestrator = create_orchestrator_node()
        tool_call = {
            "name": "test_tool",
            "args": {"input": "test"},
            "id": "call_1",
        }
        messages = [
            AIMessage(content="Using tool", tool_calls=[tool_call]),
            HumanMessage(content="Some response"),
            AIMessage(content="Another tool call", tool_calls=[tool_call]),
        ]
        state = AgentGraphState(messages=messages, current_tool_call_index=None)

        result = orchestrator(state)

        assert result == {"current_tool_call_index": 0}
