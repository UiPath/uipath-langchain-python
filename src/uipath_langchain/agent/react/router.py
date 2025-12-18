"""Routing functions for conditional edges in the agent graph."""

from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, ToolCall
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from ..exceptions import AgentNodeRoutingException
from .types import AgentGraphNode, AgentGraphState
from .utils import count_consecutive_thinking_messages

FLOW_CONTROL_TOOLS = [END_EXECUTION_TOOL.name, RAISE_ERROR_TOOL.name]


def __filter_control_flow_tool_calls(
    tool_calls: list[ToolCall],
) -> list[ToolCall]:
    """Remove control flow tools when multiple tool calls exist."""
    if len(tool_calls) <= 1:
        return tool_calls

    return [tc for tc in tool_calls if tc.get("name") not in FLOW_CONTROL_TOOLS]


def __has_control_flow_tool(tool_calls: list[ToolCall]) -> bool:
    """Check if any tool call is of a control flow tool."""
    return any(tc.get("name") in FLOW_CONTROL_TOOLS for tc in tool_calls)


def __validate_last_message_is_AI(messages: list[AnyMessage]) -> AIMessage:
    """Validate and return last message from state.

    Raises:
        AgentNodeRoutingException: If messages are empty or last message is not AIMessage
    """
    if not messages:
        raise AgentNodeRoutingException(
            "No messages in state - cannot route after agent"
        )

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        raise AgentNodeRoutingException(
            f"Last message is not AIMessage (type: {type(last_message).__name__}) - cannot route after agent"
        )

    return last_message


def create_route_agent(thinking_messages_limit: int = 0):
    """Create a routing function configured with thinking_messages_limit.

    Args:
        thinking_messages_limit: Max consecutive thinking messages before error

    Returns:
        Routing function for LangGraph conditional edges
    """

    def route_agent(
        state: AgentGraphState,
    ) -> list[str] | Literal[AgentGraphNode.AGENT, AgentGraphNode.TERMINATE]:
        """Route after agent: handles all routing logic including control flow detection.

        Routing logic:
        1. If multiple tool calls exist, filter out control flow tools (EndExecution, RaiseError)
        2. If control flow tool(s) remain, route to TERMINATE
        3. If regular tool calls remain, route to specific tool nodes (return list of tool names)
        4. If no tool calls, handle consecutive completions

        Returns:
            - list[str]: Tool node names for parallel execution
            - AgentGraphNode.AGENT: For consecutive completions
            - AgentGraphNode.TERMINATE: For control flow termination

        Raises:
            AgentNodeRoutingException: When encountering unexpected state (empty messages, non-AIMessage, or excessive completions)
        """
        messages = state.messages
        last_message = __validate_last_message_is_AI(messages)

        tool_calls = list(last_message.tool_calls) if last_message.tool_calls else []
        tool_calls = __filter_control_flow_tool_calls(tool_calls)

        if tool_calls and __has_control_flow_tool(tool_calls):
            return AgentGraphNode.TERMINATE

        if tool_calls:
            return [tc["name"] for tc in tool_calls]

        consecutive_thinking_messages = count_consecutive_thinking_messages(messages)

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

    return route_agent
