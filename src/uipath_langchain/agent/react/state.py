from __future__ import annotations

from enum import StrEnum

from langgraph.graph import MessagesState


class AgentGraphState(MessagesState):
    """Agent Graph state for standard loop execution."""

    pass


class AgentGraphNode(StrEnum):
    INIT = "init"
    AGENT = "agent"
    TOOLS = "tools"
    TERMINATE = "terminate"
