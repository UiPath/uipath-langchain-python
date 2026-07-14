"""State types for the advanced agent wrapper graph."""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AdvancedAgentGraphState(BaseModel):
    """Graph state for the advanced agent wrapper."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    structured_response: dict[str, Any] = Field(default_factory=dict)


class ConversationalAdvancedAgentGraphState(BaseModel):
    """Graph state for the conversational advanced agent wrapper."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    initial_message_count: int | None = None
