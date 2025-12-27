from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing import TypedDict

class State(TypedDict, total=False):
    branch_a_first_result: str | None
    branch_a_second_result: str | None
    branch_b_first_result: str | None
    branch_b_second_result: str | None

def branch_a_first(state: State) -> State:
    result = interrupt({"message": "Branch A - First interrupt"})
    return {"branch_a_first_result": f"A-1 completed with: {result}"}

def branch_a_second(state: State) -> State:
    result = interrupt({"message": "Branch A - Second interrupt"})
    return {"branch_a_second_result": f"A-2 completed with: {result}"}

def branch_b_first(state: State) -> State:
    result = interrupt({"message": "Branch B - First interrupt"})
    return {"branch_b_first_result": f"B-1 completed with: {result}"}

def branch_b_second(state: State) -> State:
    result = interrupt({"message": "Branch B - Second interrupt"})
    return {"branch_b_second_result": f"B-2 completed with: {result}"}

builder = StateGraph(State)
builder.add_node("branch_a_first", branch_a_first)
builder.add_node("branch_a_second", branch_a_second)
builder.add_node("branch_b_first", branch_b_first)
builder.add_node("branch_b_second", branch_b_second)

builder.add_edge(START, "branch_a_first")
builder.add_edge("branch_a_first", "branch_a_second")
builder.add_edge("branch_a_second", END)

builder.add_edge(START, "branch_b_first")
builder.add_edge("branch_b_first", "branch_b_second")
builder.add_edge("branch_b_second", END)

graph = builder.compile()
