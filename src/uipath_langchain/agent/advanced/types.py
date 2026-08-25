"""State types for the advanced agent wrapper graph."""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class AdvancedAgentGraphState(BaseModel):
    """Graph state for the advanced agent wrapper."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    structured_response: dict[str, Any] = {}


class _ConversationalAdvancedAgentGraphInput(BaseModel):
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


class ConversationalAdvancedAgentGraphState(_ConversationalAdvancedAgentGraphInput):
    """Graph state for the conversational advanced agent wrapper."""

    initial_message_count: int | None = None
