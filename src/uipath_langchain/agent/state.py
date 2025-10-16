from enum import Enum

from langgraph.graph import MessagesState


class LowCodeAgentState(MessagesState):
    """LowCodeAgentState extending MessagesState"""


class GraphNode(str, Enum):
    """Graph node identifiers for the LowCode agent graph."""

    STATE_INIT = "state_init"
    AGENT = "agent"
    TOOLS = "tools"
    TERMINATE = "terminate"
