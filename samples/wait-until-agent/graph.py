from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from uipath.platform.common import WaitUntil


class State(TypedDict, total=False):
    message: str
    resumed_at: str
    resume_payload: str


def wait_until_node(state: State) -> State:
    resume_time = datetime.now(timezone.utc) + timedelta(minutes=5)
    resume_payload = interrupt(WaitUntil(resume_time=resume_time))

    return {
        "message": state.get("message", "wait completed"),
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "resume_payload": str(resume_payload),
    }


builder = StateGraph(State)
builder.add_node("wait_until", wait_until_node)
builder.add_edge(START, "wait_until")
builder.add_edge("wait_until", END)

graph = builder.compile()
