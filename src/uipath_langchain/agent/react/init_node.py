"""State initialization node for the ReAct Agent graph."""

from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentGraphState


def create_init_node(
    messages: Sequence[SystemMessage | HumanMessage],
):
    def graph_state_init(_: AgentGraphState):
        return {"messages": list(messages)}

    return graph_state_init
