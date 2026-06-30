from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from uipath.platform.common import InvokeProcess, WaitUntil

CHILD_PROCESS_NAME = "timeout-child-agent"
PROCESS_FOLDER_PATH = "Shared"


class ParentState(TypedDict, total=False):
    message: str
    status: str
    child_result: str


def parent_node(state: ParentState) -> ParentState:
    child_result = interrupt(
        [
            InvokeProcess(
                name=CHILD_PROCESS_NAME,
                process_folder_path=PROCESS_FOLDER_PATH,
                input_arguments={"message": state.get("message", "start child work")},
            ),
            WaitUntil(
                # allows up to 10 minutes for the child process to finish.
                resume_time=datetime.now(timezone.utc) + timedelta(minutes=10),
            ),
        ]
    )

    if isinstance(child_result, dict) and "resumeTime" in child_result:
        raise TimeoutError("Child process did not finish before the timer fired.")

    return {
        "status": "completed",
        "child_result": str(child_result),
    }


parent_builder = StateGraph(ParentState)
parent_builder.add_node("parent", parent_node)
parent_builder.add_edge(START, "parent")
parent_builder.add_edge("parent", END)
parent_graph = parent_builder.compile()
