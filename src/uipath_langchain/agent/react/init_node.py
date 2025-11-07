"""State initialization node for the ReAct Agent graph."""

from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage


def create_init_node(
    messages: Sequence[SystemMessage | HumanMessage]
    | Callable[[Any], Sequence[SystemMessage | HumanMessage]],
):
    def graph_state_init(state: Any):
        if callable(messages):
            resolved_messages = messages(state)
        else:
            resolved_messages = messages

        return {"messages": list(resolved_messages)}

    return graph_state_init
