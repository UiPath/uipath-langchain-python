"""ReAct Agent loop utilities."""

from typing import Any, Sequence, TypeVar, cast

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL

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
