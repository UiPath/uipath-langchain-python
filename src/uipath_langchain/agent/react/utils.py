"""ReAct Agent loop utilities."""

from typing import Any, Sequence, TypeVar, cast

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL

from uipath_langchain.agent.exceptions import AgentStateException
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import (
    AgentGraphState,
    AgentGuardrailsGraphState,
)


def resolve_input_model(
    input_schema: dict[str, Any] | None,
) -> type[BaseModel]:
    """Resolve the input model from the input schema."""
    if input_schema:
        return create_model(input_schema)

    return BaseModel


def resolve_output_model(
    output_schema: dict[str, Any] | None,
) -> type[BaseModel]:
    """Fallback to default end_execution tool schema when no agent output schema is provided."""
    if output_schema:
        return create_model(output_schema)

    return END_EXECUTION_TOOL.args_schema


def extract_input_data_from_state(
    state: BaseModel | dict[str, Any],
    input_model: type[BaseModel],
) -> dict[str, Any]:
    """Extract only input schema fields from graph state, filtering out internal fields.

    This prevents internal LangGraph state fields (messages, termination, agent_outcome, etc.)
    from leaking into template interpolation.

    Args:
        state: The combined agent graph state (InnerAgentGraphState = AgentGraphState + input_schema).
               At runtime, this is a dynamically created class that inherits from both.
        input_model: The input schema model defining allowed fields

    Returns:
        Dictionary containing only graph input arguments defined in the agent's input_schema
    """
    if isinstance(state, BaseModel):
        graph_state = state.model_dump()
    else:
        graph_state = state
    return input_model.model_validate(graph_state, from_attributes=True).model_dump()


def count_consecutive_thinking_messages(messages: Sequence[BaseMessage]) -> int:
    """Count consecutive AIMessages without tool calls at end of message history."""
    if not messages:
        return 0

    count = 0
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            break

        if message.tool_calls:
            break

        if not message.content:
            break

        count += 1

    return count


InputT = TypeVar("InputT", bound=BaseModel)
GraphStateT = TypeVar("GraphStateT", bound=BaseModel)


def _create_state_model_with_input(
    state_model: type[GraphStateT],
    input_schema: type[InputT] | None,
    model_name: str = "CompleteStateModel",
) -> type[GraphStateT]:
    if input_schema is None:
        return state_model

    CompleteStateModel = type(
        model_name,
        (state_model, input_schema),
        {},
    )

    cast(type[GraphStateT], CompleteStateModel).model_rebuild()
    return CompleteStateModel


def create_state_with_input(input_schema: type[InputT] | None) -> type[AgentGraphState]:
    return _create_state_model_with_input(
        AgentGraphState, input_schema, model_name="CompleteAgentGraphState"
    )


def create_guardrails_state_with_input(
    input_schema: type[InputT] | None,
) -> type[AgentGuardrailsGraphState]:
    return _create_state_model_with_input(
        AgentGuardrailsGraphState,
        input_schema,
        model_name="CompleteAgentGuardrailsGraphState",
    )


def find_latest_ai_message(messages: list[AnyMessage]) -> AIMessage | None:
    """Find and return the latest AIMessage from a list of messages.

    Args:
        messages: List of messages to search through

    Returns:
        The latest AIMessage found, or None if no AIMessage exists
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def extract_current_tool_call_index(
    messages: list[AnyMessage], tool_name: str | None = None
) -> int | None:
    """Extract the current tool_call_index from messages list.

    Iterates messages in reverse to find tool calls that already received responses,
    then looks at the last AIMessage to determine the next tool call index.

    Args:
        messages: List of messages to analyze
        tool_name: If provided, only consider tool calls for this specific tool

    Returns:
        int: The index of the next tool call to execute (0-based)
        None: If all tool calls have been completed
    """
    completed_tool_call_ids = set()

    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            completed_tool_call_ids.add(message.tool_call_id)
        elif isinstance(message, AIMessage):
            if not message.tool_calls:
                raise AgentStateException(
                    "No tool calls found in latest AIMessage while extracting tool call index."
                )

            # find the first tool call that hasn't been completed
            for i, tool_call in enumerate(message.tool_calls):
                if tool_name is None or tool_call["name"] == tool_name:
                    if tool_call["id"] not in completed_tool_call_ids:
                        return i

            # all tool calls completed
            return None
        else:
            raise AgentStateException(
                f"Unexpected message type {type(message)} encountered while extracting tool call index. Expected AIMessage or ToolMessage."
            )

    raise AgentStateException(
        "No AIMessage found in messages - cannot extract current tool call index"
    )
