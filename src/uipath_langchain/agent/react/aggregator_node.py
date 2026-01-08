"""Aggregator node for merging substates back into main state."""

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.types import Overwrite

from uipath_langchain.agent.react.types import AgentGraphState, InnerAgentGraphState


def _aggregate_messages(
    original_messages: list[AnyMessage], substate_messages: dict[str, list[AnyMessage]]
) -> list[AnyMessage]:
    aggregated_by_id: dict[str, AnyMessage] = {}
    original_order: list[str] = []

    for msg in original_messages:
        aggregated_by_id[msg.id] = msg
        original_order.append(msg.id)

    new_messages: list[AnyMessage] = []

    for tool_call_id, substate_msgs in substate_messages.items():
        for msg in substate_msgs:
            if msg.id in aggregated_by_id:
                # existing message
                original_msg = aggregated_by_id[msg.id]
                if (
                    isinstance(msg, AIMessage)
                    and msg.tool_calls
                    and len(msg.tool_calls) > 0
                ):
                    updated_tool_call = next(
                        (tc for tc in msg.tool_calls if tc["id"] == tool_call_id), None
                    )
                    if updated_tool_call:
                        # update the specific tool call in the original message
                        new_tool_calls = [
                            updated_tool_call if tc["id"] == tool_call_id else tc
                            for tc in original_msg.tool_calls
                        ]
                        aggregated_by_id[msg.id].tool_calls = new_tool_calls
            else:
                # new message, add it
                new_messages.append(msg)

    result = []
    for msg_id in original_order:
        result.append(aggregated_by_id[msg_id])
    result.extend(new_messages)

    return result


def create_aggregator_node() -> callable:
    """Create an aggregator node that merges substates back into main state."""

    def aggregator_node(state: AgentGraphState) -> dict[str, Any] | Overwrite:
        """
        Aggregate substates back into main state.

        If substates is empty, no-op and continue.
        If substates is non-empty:
        - for messages, leave placeholder for message aggregation logic
        - for each field in inner state, get its reducer and apply updates
        - lastly, overwrite the state and clear substates
        """
        if not state.substates:
            return {}

        # message aggregation
        substate_messages = {}
        for tool_call_id, substate in state.substates.items():
            if "messages" in substate:
                substate_messages[tool_call_id] = substate["messages"]

        aggregated_messages = _aggregate_messages(state.messages, substate_messages)

        # inner state fields aggregation
        aggregated_inner_dict = state.inner_state.model_dump()

        inner_state_fields = InnerAgentGraphState.model_fields
        for substate in state.substates.values():
            if "inner_state" in substate:
                substate_inner_data = substate["inner_state"]

                if isinstance(substate_inner_data, InnerAgentGraphState):
                    substate_inner_dict = substate_inner_data.model_dump()
                else:
                    substate_inner_dict = substate_inner_data

                # for each field, apply reducer if defined
                for field_name, field_info in inner_state_fields.items():
                    if field_name in substate_inner_dict:
                        substate_field_value = substate_inner_dict[field_name]
                        current_field_value = aggregated_inner_dict[field_name]

                        if field_info.metadata and callable(field_info.metadata[-1]):
                            reducer_func = field_info.metadata[-1]
                            merged_value = reducer_func(
                                current_field_value, substate_field_value
                            )
                        else:
                            # no reducer, just replace
                            merged_value = substate_field_value

                        aggregated_inner_dict[field_name] = merged_value

        aggregated_inner_state = InnerAgentGraphState.model_validate(
            aggregated_inner_dict
        )

        state.messages = aggregated_messages
        state.inner_state = aggregated_inner_state
        state.substates = {}

        # return overwrite command to replace the state
        return {
            **state.model_dump(exclude={"messages", "inner_state", "substates"}),
            "messages": Overwrite(aggregated_messages),
            "inner_state": Overwrite(aggregated_inner_state),
            "substates": Overwrite({}),
        }

    return aggregator_node
