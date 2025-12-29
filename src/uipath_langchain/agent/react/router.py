"""Routing functions for conditional edges in the agent graph."""

from typing import Literal

from ..exceptions import AgentNodeRoutingException
from .types import (
    FLOW_CONTROL_TOOLS,
    AgentGraphNode,
    AgentGraphState,
)
from .utils import find_latest_ai_message


def create_route_agent():
    """Create a routing function for LangGraph conditional edges.

    Returns:
        Routing function for LangGraph conditional edges
    """

    def route_agent(
        state: AgentGraphState,
    ) -> str | Literal[AgentGraphNode.AGENT, AgentGraphNode.TERMINATE]:
        """Route after agent: looks at current tool call index and routes to corresponding tool node.

        Routing logic:
        1. If current_tool_call_index is None, route back to LLM
        2. If current_tool_call_index is set, route to the corresponding tool node
        3. Handle control flow tools for termination

        Returns:
            - str: Tool node name for single tool execution
            - AgentGraphNode.AGENT: When no current tool call index
            - AgentGraphNode.TERMINATE: For control flow termination

        Raises:
            AgentNodeRoutingException: When encountering unexpected state
        """
        current_index = state.current_tool_call_index

        # no tool call in progress, route back to LLM
        if current_index is None:
            return AgentGraphNode.AGENT

        messages = state.messages

        if not messages:
            raise AgentNodeRoutingException(
                "No messages in state - cannot route after agent"
            )

        latest_ai_message = find_latest_ai_message(messages)

        if latest_ai_message is None:
            raise AgentNodeRoutingException(
                "No AIMessage found in messages - cannot route after agent"
            )

        tool_calls = (
            list(latest_ai_message.tool_calls) if latest_ai_message.tool_calls else []
        )

        if current_index >= len(tool_calls):
            raise AgentNodeRoutingException(
                f"Current tool call index {current_index} exceeds available tool calls ({len(tool_calls)})"
            )

        current_tool_call = tool_calls[current_index]
        tool_name = current_tool_call["name"]

        # handle control flow tools for termination
        if tool_name in FLOW_CONTROL_TOOLS:
            return AgentGraphNode.TERMINATE

        return tool_name

    return route_agent
