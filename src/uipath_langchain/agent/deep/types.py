"""State types for the deep agent wrapper graph."""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class DeepAgentGraphState(BaseModel):
    """Graph state for the deep agent wrapper."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    structured_response: dict[str, Any] = {}
