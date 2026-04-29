from pydantic import BaseModel

from langgraph.graph import END, START, StateGraph


class Score(BaseModel):
    label: str = ""
    value: float = 0.0


class State(BaseModel):
    topic: str
    score: Score | None = None


async def evaluate(state: State) -> State:
    return State(topic=state.topic, score=Score(label="ok", value=1.0))


async def finalize(state: State) -> State:
    return state


builder = StateGraph(State)
builder.add_node("evaluate", evaluate)
builder.add_node("finalize", finalize)
builder.add_edge(START, "evaluate")
builder.add_edge("evaluate", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
