"""Utilities for the deep agent wrapper graph."""

from typing import cast

from pydantic import BaseModel

from .types import DeepAgentGraphState


def create_state_with_input(
    input_schema: type[BaseModel] | None,
) -> type[DeepAgentGraphState]:
    """Create combined state by merging DeepAgentGraphState with the input schema.

    Mirrors the shallow agent's create_state_with_input pattern:
    dynamic multi-inheritance + model_rebuild() for Pydantic resolution.
    """
    if input_schema is None:
        return DeepAgentGraphState
    CompleteState = type(
        "CompleteDeepAgentGraphState",
        (DeepAgentGraphState, input_schema),
        {},
    )
    cast(type[BaseModel], CompleteState).model_rebuild()
    return CompleteState
