import random
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict
from uipath.models import CreateAction


# State
class State(TypedDict):
    graph_state: str


# Nodes
def node_1(state):
    print("---Node 1---")
    simple_interrupt = interrupt("question: Who are you?")

    return {"graph_state": "Hello, I am " + simple_interrupt["answer"] + "!"}


def node_2(state):
    print("---Node 2---")
    action_interrupt = interrupt(
        CreateAction(
            app_name="Test-app", title="Test-title", description="Test-description"
        )
    )
    return {"graph_state": state["graph_state"] + action_interrupt["ActionData"]}


def node_3(state):
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " end"}


builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)


memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
