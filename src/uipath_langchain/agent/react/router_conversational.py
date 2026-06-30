"""Routing functions for conditional edges in the agent graph."""

import logging
from collections.abc import Container
from typing import Literal

from uipath.agent.react import END_EXECUTION_TOOL
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


def create_route_agent_conversational(valid_targets: Container[str] | None = None):
    """Create a routing function for conversational agents.

    Routes between AGENT, tool nodes, and TERMINATE based on the latest AIMessage's
    tool calls. Conversational agents terminate either by emitting an AIMessage with
    no tool calls (the model has decided the turn is complete) or by calling a
    flow-control tool (`end_execution`).

    Args:
        valid_targets: Allowed routing destinations.
    Returns:
        Routing function for LangGraph conditional edges.
    """

    def route_agent_conversational(
        state: AgentGraphState,
    ) -> str | Literal[AgentGraphNode.TERMINATE] | Literal[AgentGraphNode.AGENT]:
        """Routing logic for the conversational agent.

        Routing logic:
        1. No AIMessage in state → ROUTING_ERROR (SYSTEM).
        2. Latest AIMessage carries an `end_execution` tool call → TERMINATE
           (agent finalized its turn and provided custom output args).
        3. Latest AIMessage has no tool calls → TERMINATE (agent finalized its
           turn without a custom output schema).
        4. Latest AIMessage has tool calls and at least one is still pending a
           ToolMessage → route to that tool's node.
        5. Latest AIMessage has tool calls and all have been answered → AGENT
           (loop back so the model can continue its turn).

        Returns:
            - str: Tool node name to route to.
            - AgentGraphNode.AGENT: Loop back to the model after all tools have responded.
            - AgentGraphNode.TERMINATE: Turn complete.

        Raises:
            AgentRuntimeError: ROUTING_ERROR when state has no AIMessage, or when
                a routed tool name is not in `valid_targets`.
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
            # `end_execution` signals the agent has finalized its turn (carries the
            # structured output args).
            if any(
                tc["name"] == END_EXECUTION_TOOL.name for tc in last_message.tool_calls
            ):
                return AgentGraphNode.TERMINATE

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
            return AgentGraphNode.TERMINATE

    return route_agent_conversational
