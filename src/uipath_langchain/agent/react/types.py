from __future__ import annotations

from enum import StrEnum

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class AgentGraphState(MessagesState):
    """Agent Graph state for standard loop execution."""

    pass


class AgentGraphNode(StrEnum):
    INIT = "init"
    AGENT = "agent"
    TOOLS = "tools"
    TERMINATE = "terminate"


class AgentGraphConfig(BaseModel):
    recursion_limit: int = Field(
        default=50, ge=1, description="Maximum recursion limit for the agent graph"
    )
