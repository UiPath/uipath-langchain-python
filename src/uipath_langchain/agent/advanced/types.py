"""State types for the advanced agent wrapper graph."""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AdvancedAgentGraphState(BaseModel):
    """Graph state for the advanced agent wrapper."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    structured_response: dict[str, Any] = {}
