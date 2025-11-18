from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentGraphState(BaseModel):
    """Agent Graph state for standard loop execution."""

    messages: Annotated[list[AnyMessage], add_messages] = []


class AgentGraphNode(StrEnum):
    INIT = "init"
    AGENT = "agent"
    TOOLS = "tools"
    TERMINATE = "terminate"


class AgentGraphConfig(BaseModel):
    recursion_limit: int = Field(
        default=50, ge=1, description="Maximum recursion limit for the agent graph"
    )
