"""Routing functions for conditional edges in the agent graph."""

from collections.abc import Container
from typing import Literal

from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from .types import FLOW_CONTROL_TOOLS, AgentGraphNode, AgentGraphState
from .utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)


def create_route_agent(
    valid_targets: Container[str] | None = None,
):
    """Create the conditional-edge routing function.

    Args:
        valid_targets: Allowed routing destinations

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

        A tool-less turn with content always loops back to AGENT: the router can't
        tell whether tool_choice was actually forced on the wire (handlers silently
        downgrade it under thinking), so the LLM node owns stall accounting —
        forcing, the extraction retry, and the deterministic failure.

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
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.ROUTING_ERROR,
                title="No AIMessage found in messages.",
                detail="The agent state contains no AIMessage, which is required for routing decisions.",
                category=UiPathErrorCategory.SYSTEM,
            )

        if not last_message.tool_calls:
            if last_message.content:
                return AgentGraphNode.AGENT

            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.ROUTING_ERROR,
                title="Agent produced empty response without tool calls.",
                detail="The model returned no content and no tool calls. "
                "If you are using a BYOM configuration, verify your model deployment.",
                category=UiPathErrorCategory.SYSTEM,
            )

        current_index = extract_current_tool_call_index(messages)

        # all tool calls completed, go back to agent
        if current_index is None:
            return AgentGraphNode.AGENT

        current_tool_call = last_message.tool_calls[current_index]
        current_tool_name = current_tool_call["name"]

        if current_tool_name in FLOW_CONTROL_TOOLS:
            return AgentGraphNode.TERMINATE

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

    return route_agent
