"""Routing functions for conditional edges in the agent graph."""

import logging
from typing import Literal

from uipath_langchain.agent.react.router_utils import validate_last_message_is_AI

from .types import AgentGraphNode, AgentGraphState

logger = logging.getLogger(__name__)


def create_route_agent_conversational():
    """Create a routing function for conversational agents. It routes between agent and tool calls until
    the agent response has no tool calls, then it routes to the USER_MESSAGE_WAIT node which does an interrupt.

    Returns:
        Routing function for LangGraph conditional edges
    """

    def route_agent_conversational(
        state: AgentGraphState,
    ) -> list[str] | Literal[AgentGraphNode.TERMINATE]:
        """Route after agent

        Routing logic:
        3. If tool calls, route to specific tool nodes (return list of tool names)
        4. If no tool calls, route to user message wait node

        Returns:
            - list[str]: Tool node names for parallel execution
            - AgentGraphNode.USER_MESSAGE_WAIT: When there are no tool calls

        Raises:
            AgentNodeRoutingException: When encountering unexpected state (empty messages, non-AIMessage, or excessive completions)
        """
        last_message = validate_last_message_is_AI(state.messages)
        if last_message.tool_calls:
            return [tc["name"] for tc in last_message.tool_calls]
        else:
            return AgentGraphNode.TERMINATE

    return route_agent_conversational
