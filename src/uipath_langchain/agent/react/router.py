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


def filter_control_flow_tool_calls_from_state(state: AgentGraphState) -> AgentGraphState:
    """Remove filtered control flow tool calls from AIMessage to prevent OpenAI API errors.

    When multiple tools are called and one is a control flow tool (end_execution, raise_error),
    the control flow tools are filtered out for execution. However, the AIMessage still
    contains these tool calls, causing OpenAI to expect ToolMessages for them.

    This node updates the AIMessage to only include tool calls that will actually be executed.
    """
    messages = state.messages
    if not messages:
        return state

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return state

    original_tool_calls = list(last_message.tool_calls)
    if len(original_tool_calls) <= 1:
        # No filtering needed for single tool calls
        return state

    # Check if any control flow tools would be filtered
    has_control_flow = any(
        tc.get("name") in FLOW_CONTROL_TOOLS for tc in original_tool_calls
    )
    if not has_control_flow:
        # No control flow tools to filter
        return state

    # Filter out control flow tools
    filtered_tool_calls = [
        tc for tc in original_tool_calls if tc.get("name") not in FLOW_CONTROL_TOOLS
    ]

    if len(filtered_tool_calls) == len(original_tool_calls):
        # No filtering occurred
        return state

    # Filter content if it's a list of tool call dicts
    filtered_content = last_message.content
    if isinstance(last_message.content, list):
        # Filter out control flow tools from content as well
        filtered_ids = {tc["id"] for tc in filtered_tool_calls}
        filtered_content = [
            item
            for item in last_message.content
            if not (
                isinstance(item, dict)
                and item.get("type") == "function_call"
                and item.get("call_id") not in filtered_ids
            )
        ]

    # Create new AIMessage with only non-control-flow tool calls
    updated_message = AIMessage(
        content=filtered_content,
        tool_calls=filtered_tool_calls,
        id=last_message.id,
        additional_kwargs=last_message.additional_kwargs,
        response_metadata=last_message.response_metadata,
        usage_metadata=last_message.usage_metadata,
    )

    # Return updated state with modified message
    updated_messages = list(messages[:-1]) + [updated_message]
    return AgentGraphState(
        messages=updated_messages,
        inner_state=state.inner_state,
    )


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
