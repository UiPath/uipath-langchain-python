from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from uipath.platform.common import WaitTimeTrigger


class State(TypedDict):
    message: str


def main_node(state: State) -> State:
    response = interrupt(
        WaitTimeTrigger(
            cron_expression="0 0/5 * * * ?",
            time_zone_id="Europe/Bucharest",
        )
    )
    return {"message": str(response)}


builder: StateGraph[State] = StateGraph(State)

builder.add_node("main_node", main_node)

builder.add_edge(START, "main_node")
builder.add_edge("main_node", END)


memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
