from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath.platform.common import InvokeProcess, WaitUntil, assert_no_timeout

CHILD_PROCESS_NAME = "timeout-child-agent"
CHILD_PROCESS_FOLDER_PATH = "Shared"


class Input(BaseModel):
    message: str = "run the child process"


class Output(BaseModel):
    result: str


class State(BaseModel):
    message: str
    result: str = ""


async def parent_node(state: State) -> dict[str, Any]:
    resume_time = datetime.now(UTC) + timedelta(minutes=10)

    child_result = interrupt(
        [
            InvokeProcess(
                name=CHILD_PROCESS_NAME,
                process_folder_path=CHILD_PROCESS_FOLDER_PATH,
                input_arguments={"message": state.message},
            ),
            # allows up to 10 minutes for the child process to finish.
            WaitUntil(resume_time=resume_time),
        ]
    )

    # raises UiPathTimeoutError on timeout.
    assert_no_timeout(child_result)

    return {
        "message": state.message,
        "result": f"child completed: {child_result}",
    }


builder = StateGraph(state_schema=State)
builder.add_node("parent", parent_node)
builder.add_edge(START, "parent")
builder.add_edge("parent", END)

graph = builder.compile()
