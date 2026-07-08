"""Routing functions for conditional edges in the agent graph."""

import logging
from collections.abc import Container
from typing import Literal

from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)

from .types import AgentGraphNode

logger = logging.getLogger(__name__)


def create_route_agent_conversational(
    valid_targets: Container[str] | None = None,
    with_generate_output_node: bool = False,
):
    """Create a routing function for conversational agents. It routes between agent and tool calls until
    the agent response has no tool calls, then it routes to either the
    GENERATE_CONVERSATIONAL_OUTPUT node (when the agent declares custom output
    fields) or directly to TERMINATE.

    Args:
        valid_targets: Allowed routing destinations.
        with_generate_output_node: When True, route AGENT-without-tool-calls to the
            GENERATE_CONVERSATIONAL_OUTPUT node so the structured output can
            be extracted before TERMINATE. When False, route straight to
            TERMINATE.
    Returns:
        Routing function for LangGraph conditional edges
    """

    def route_agent_conversational(
        state: AgentGraphState,
    ) -> (
        str
        | Literal[AgentGraphNode.TERMINATE]
        | Literal[AgentGraphNode.AGENT]
        | Literal[AgentGraphNode.GENERATE_CONVERSATIONAL_OUTPUT]
    ):
        """Route after agent.

        Routing logic:
        - If the latest AIMessage has tool calls
            - If pending tools, route to the next pending tool node.
            - Otherwise: route to AGENT as all tool calls completed.
        - Otherwise:
            - If schema declares custom output fields: route to
            GENERATE_CONVERSATIONAL_OUTPUT to generate the output fields.
            - Otherwise: route straight to TERMINATE.

        Raises:
            AgentRuntimeError: ROUTING_ERROR when state has no AIMessage, or
                when a routed tool name is not in `valid_targets`.
        """
        last_message = find_latest_ai_message(state.messages)
        if last_message is None:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.ROUTING_ERROR,
                title="No AIMessage found in messages for routing.",
                detail="The agent state contains no AIMessage, which is required for routing decisions.",
                category=UiPathErrorCategory.SYSTEM,
            )
        if last_message.tool_calls:
            current_index = extract_current_tool_call_index(state.messages)
            # all tool calls completed, go back to agent
            if current_index is None:
                return AgentGraphNode.AGENT

            current_tool_call = last_message.tool_calls[current_index]
            current_tool_name = current_tool_call["name"]

            if valid_targets is not None and current_tool_name not in valid_targets:
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.ROUTING_ERROR,
                    title="Agent routed to an unknown destination",
                    detail=(
                        f"The agent attempted to route to '{current_tool_name}', "
                        "which is not a registered tool or control node."
                    ),
                    category=UiPathErrorCategory.SYSTEM,
                )

            return current_tool_name
        else:
            return (
                AgentGraphNode.GENERATE_CONVERSATIONAL_OUTPUT
                if with_generate_output_node
                else AgentGraphNode.TERMINATE
            )

    return route_agent_conversational
