from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from uipath.platform.common import InvokeProcess
from uipath.platform.resume_triggers import assert_no_timeout

CHILD_PROCESS_NAME = "timeout-child-agent"
PROCESS_FOLDER_PATH = "Shared"


class ParentState(TypedDict, total=False):
    message: str
    status: str
    child_result: str
    timeout: dict[str, object]


def parent_node(state: ParentState) -> ParentState:
    child_result = interrupt(
        InvokeProcess(
            name=CHILD_PROCESS_NAME,
            process_folder_path=PROCESS_FOLDER_PATH,
            input_arguments={"message": state.get("message", "start child work")},
            # allows up to 10 minutes for the child process to finish.
            timeout=600,
        )
    )

    # Raises UiPathTimeoutError on timeout.
    assert_no_timeout(child_result)

    return {
        "status": "completed",
        "child_result": str(child_result),
    }


parent_builder = StateGraph(ParentState)
parent_builder.add_node("parent", parent_node)
parent_builder.add_edge(START, "parent")
parent_builder.add_edge("parent", END)
parent_graph = parent_builder.compile()
