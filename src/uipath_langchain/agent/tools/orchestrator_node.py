from typing import Any

from langchain_core.messages import ToolCall

from uipath_langchain.agent.exceptions import AgentNodeRoutingException
from uipath_langchain.agent.react.types import FLOW_CONTROL_TOOLS, AgentGraphState
from uipath_langchain.agent.react.utils import (
    count_consecutive_thinking_messages,
    find_latest_ai_message,
)


def __filter_control_flow_tool_calls(tool_calls: list[ToolCall]) -> list[ToolCall]:
    """Remove control flow tools when multiple tool calls exist."""
    if len(tool_calls) <= 1:
        return tool_calls

    return [tc for tc in tool_calls if tc.get("name") not in FLOW_CONTROL_TOOLS]


def create_orchestrator_node(thinking_messages_limit: int = 0):
    """Create an orchestrator node responsible for sequencing tool calls.

    Args:
        thinking_messages_limit: Max consecutive thinking messages before error
    """

    def orchestrator_node(state: AgentGraphState) -> dict[str, Any]:
        current_index = state.current_tool_call_index

        if current_index is None:
            # new batch of tool calls
            if not state.messages:
                raise AgentNodeRoutingException(
                    "No messages in state - cannot process tool calls"
                )

            # check consecutive thinking messages limit
            if thinking_messages_limit >= 0:
                consecutive_thinking = count_consecutive_thinking_messages(
                    state.messages
                )
                if consecutive_thinking > thinking_messages_limit:
                    raise AgentNodeRoutingException(
                        f"Too many consecutive thinking messages ({consecutive_thinking}). "
                        f"Limit is {thinking_messages_limit}. Agent must use tools."
                    )

            latest_ai_message = find_latest_ai_message(state.messages)

            if latest_ai_message is None or not latest_ai_message.tool_calls:
                return {"current_tool_call_index": None}

            # apply flow control tool filtering
            original_tool_calls = list(latest_ai_message.tool_calls)
            filtered_tool_calls = __filter_control_flow_tool_calls(original_tool_calls)

            if len(filtered_tool_calls) != len(original_tool_calls):
                modified_message = latest_ai_message.model_copy()
                modified_message.tool_calls = filtered_tool_calls

                # we need to filter out the content within the message as well, otherwise the LLM will raise an error
                filtered_tool_call_ids = {tc["id"] for tc in filtered_tool_calls}
                if isinstance(modified_message.content, list):
                    modified_message.content = [
                        block
                        for block in modified_message.content
                        if (
                            isinstance(block, dict)
                            and (
                                block.get("call_id") in filtered_tool_call_ids
                                or block.get("call_id") is None  # keep non-tool blocks
                            )
                        )
                        or not isinstance(block, dict)
                    ]

                return {
                    "current_tool_call_index": 0,
                    "messages": [modified_message],
                }

            return {"current_tool_call_index": 0}

        # in the middle of processing a batch
        if not state.messages:
            raise AgentNodeRoutingException(
                "No messages in state during batch processing"
            )

        latest_ai_message = find_latest_ai_message(state.messages)

        if latest_ai_message is None:
            raise AgentNodeRoutingException(
                "No AI message found during batch processing"
            )

        if not latest_ai_message.tool_calls:
            raise AgentNodeRoutingException(
                "No tool calls found in AI message during batch processing"
            )

        total_tool_calls = len(latest_ai_message.tool_calls)
        next_index = current_index + 1

        if next_index >= total_tool_calls:
            return {"current_tool_call_index": None}
        else:
            return {"current_tool_call_index": next_index}

    return orchestrator_node
