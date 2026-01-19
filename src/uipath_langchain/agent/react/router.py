"""Routing functions for conditional edges in the agent graph."""

from typing import Literal

from ..exceptions import AgentNodeRoutingException
from .types import FLOW_CONTROL_TOOLS, AgentGraphNode, AgentGraphState
from .utils import (
    count_consecutive_thinking_messages,
    extract_current_tool_call_index,
    find_latest_ai_message,
)


def create_route_agent(thinking_messages_limit: int = 0):
    """Create a routing function configured with thinking_messages_limit.

    Args:
        thinking_messages_limit: Max consecutive thinking messages before error

    Returns:
        Routing function for LangGraph conditional edges
    """

    def route_agent(
        state: AgentGraphState,
    ) -> str | Literal[AgentGraphNode.AGENT, AgentGraphNode.TERMINATE]:
        """Route after agent: handles sequential tool execution.

        Routing logic:
        1. Get current tool call index from messages
        2. If current tool call index is None (all tools completed), route to AGENT
        3. If current tool call is a flow control tool, route to TERMINATE
        4. Otherwise, route to the specific tool node

        Returns:
            - str: Single tool node name for sequential execution
            - AgentGraphNode.AGENT: When all tool calls completed or no tool calls
            - AgentGraphNode.TERMINATE: For control flow termination

        Raises:
            AgentNodeRoutingException: When encountering unexpected state
        """
        messages = state.messages
        last_message = find_latest_ai_message(messages)
        if last_message is None:
            raise AgentNodeRoutingException(
                "No AIMessage found in messages for routing."
            )

        if not last_message.tool_calls:
            consecutive_thinking_messages = count_consecutive_thinking_messages(
                messages
            )

            if consecutive_thinking_messages > thinking_messages_limit:
                raise AgentNodeRoutingException(
                    f"Agent exceeded consecutive completions limit without producing tool calls "
                    f"(completions: {consecutive_thinking_messages}, max: {thinking_messages_limit}). "
                    f"This should not happen as tool_choice='required' is enforced at the limit."
                )

            if last_message.content:
                return AgentGraphNode.AGENT

            raise AgentNodeRoutingException(
                f"Agent produced empty response without tool calls "
                f"(completions: {consecutive_thinking_messages}, has_content: False)"
            )

        current_index = extract_current_tool_call_index(messages)

        # all tool calls completed, go back to agent
        if current_index is None:
            return AgentGraphNode.AGENT

        current_tool_call = last_message.tool_calls[current_index]
        current_tool_name = current_tool_call["name"]

        if current_tool_name in FLOW_CONTROL_TOOLS:
            return AgentGraphNode.TERMINATE

        return current_tool_name

    return route_agent
