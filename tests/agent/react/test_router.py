"""Tests for agent routing functions."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from uipath_langchain.agent.react.exceptions import AgentNodeRoutingException
from uipath_langchain.agent.react.router import route_agent
from uipath_langchain.agent.react.types import AgentGraphNode, AgentGraphState


class TestRouteAgent:
    """Test cases for route_agent function."""

    def test_raises_on_empty_messages(self):
        """Should raise AgentNodeRoutingException when messages list is empty."""
        state = AgentGraphState(messages=[])
        with pytest.raises(AgentNodeRoutingException, match="No messages in state"):
            route_agent(state)

    def test_raises_on_non_ai_last_message(self):
        """Should raise when last message is not AIMessage."""
        state = AgentGraphState(messages=[HumanMessage(content="query")])
        with pytest.raises(AgentNodeRoutingException, match="not AIMessage"):
            route_agent(state)

    def test_routes_to_terminate_on_end_execution(self):
        """Should route to TERMINATE when end_execution tool is called."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "end_execution", "args": {}, "id": "call_1"}],
                ),
            ]
        )
        result = route_agent(state)
        assert result == AgentGraphNode.TERMINATE

    def test_routes_to_terminate_on_raise_error(self):
        """Should route to TERMINATE when raise_error tool is called."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "raise_error", "args": {}, "id": "call_1"}],
                ),
            ]
        )
        result = route_agent(state)
        assert result == AgentGraphNode.TERMINATE

    def test_routes_to_tool_nodes(self):
        """Should return list of tool names when regular tools are called."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_tool", "args": {}, "id": "call_1"},
                        {"name": "calc_tool", "args": {}, "id": "call_2"},
                    ],
                ),
            ]
        )
        result = route_agent(state)
        assert result == ["search_tool", "calc_tool"]

    def test_filters_control_flow_from_multiple_tool_calls(self):
        """Should filter out control flow tools when mixed with regular tools."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_tool", "args": {}, "id": "call_1"},
                        {"name": "end_execution", "args": {}, "id": "call_2"},
                    ],
                ),
            ]
        )
        result = route_agent(state)
        assert result == ["search_tool"]

    def test_routes_to_agent_on_text_response(self):
        """Should route back to AGENT when AI produces text without tools."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(content="thinking..."),
            ]
        )
        result = route_agent(state)
        assert result == AgentGraphNode.AGENT

    def test_raises_on_empty_ai_response_without_tools(self):
        """Should raise when AI produces empty response without tool calls."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(content=""),
            ]
        )
        with pytest.raises(AgentNodeRoutingException, match="empty response"):
            route_agent(state)

    def test_handles_single_tool_call(self):
        """Should return list with single tool name."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "my_tool", "args": {}, "id": "call_1"}],
                ),
            ]
        )
        result = route_agent(state)
        assert result == ["my_tool"]

    def test_routes_to_agent_after_tool_response(self):
        """Should route to AGENT when last message is AI text after tool use."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(
                    content="using tool",
                    tool_calls=[{"name": "tool", "args": {}, "id": "call_1"}],
                ),
                ToolMessage(content="result", tool_call_id="call_1"),
                AIMessage(content="analyzing result"),
            ]
        )
        result = route_agent(state)
        assert result == AgentGraphNode.AGENT

    def test_handles_empty_tool_calls_list(self):
        """Should treat empty tool_calls list as no tools called."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(content="response", tool_calls=[]),
            ]
        )
        result = route_agent(state)
        assert result == AgentGraphNode.AGENT

    def test_excessive_successive_completions_raises(self):
        """Should raise when too many successive completions without tool calls."""
        state = AgentGraphState(
            messages=[
                HumanMessage(content="query"),
                AIMessage(content="thinking 1"),
                AIMessage(content="thinking 2"),
                AIMessage(content="thinking 3"),
            ]
        )
        with pytest.raises(
            AgentNodeRoutingException, match="exceeded successive completions limit"
        ):
            route_agent(state)
