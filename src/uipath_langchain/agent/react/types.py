from __future__ import annotations

from enum import StrEnum
from typing import Any

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import NotRequired


class AgentGraphState(MessagesState):
    """Agent Graph state for standard loop execution."""

    output: NotRequired[dict[str, Any]]


class AgentGraphNode(StrEnum):
    INIT = "init"
    AGENT = "agent"
    TOOLS = "tools"
    TERMINATE = "terminate"


class AgentGraphConfig(BaseModel):
    recursion_limit: int = Field(
        default=50, ge=1, description="Maximum recursion limit for the agent graph"
    )
