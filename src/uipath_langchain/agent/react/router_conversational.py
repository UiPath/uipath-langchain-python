"""Routing functions for conditional edges in the agent graph."""

import logging
from typing import Literal

from uipath_langchain.agent.exceptions import AgentNodeRoutingException
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)

from .types import AgentGraphNode

logger = logging.getLogger(__name__)


def create_route_agent_conversational():
    """Create a routing function for conversational agents. It routes between agent and tool calls until
    the agent response has no tool calls, then it routes to the USER_MESSAGE_WAIT node which does an interrupt.

    Returns:
        Routing function for LangGraph conditional edges
    """

    def route_agent_conversational(
        state: AgentGraphState,
    ) -> str | Literal[AgentGraphNode.TERMINATE] | Literal[AgentGraphNode.AGENT]:
        """Route after agent

        Routing logic:
        3. If tool calls, route to specific tool nodes (return list of tool names)
        4. If no tool calls, route to user message wait node

        Returns:
            - str: Tool node name for sequential execution
            - AgentGraphNode.USER_MESSAGE_WAIT: When there are no tool calls

        Raises:
            AgentNodeRoutingException: When encountering unexpected state (empty messages, non-AIMessage, or excessive completions)
        """
        last_message = find_latest_ai_message(state.messages)
        if last_message is None:
            raise AgentNodeRoutingException(
                "No AIMessage found in messages for routing."
            )
        if last_message.tool_calls:
            current_index = extract_current_tool_call_index(state.messages)
            # all tool calls completed, go back to agent
            if current_index is None:
                return AgentGraphNode.AGENT

            current_tool_call = last_message.tool_calls[current_index]
            current_tool_name = current_tool_call["name"]

            return current_tool_name
        else:
            return AgentGraphNode.TERMINATE

    return route_agent_conversational
