# type: ignore
import dataclasses
import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from uipath.platform.action_center import Task
from uipath.platform.common import CreateTask, WaitTask


@dataclasses.dataclass
class Input:
    pass


class GraphState(BaseModel):
    message: str | None = None


@dataclasses.dataclass
class Output:
    message: str


def create_action_node(input: Input) -> Command:
    response = interrupt(
        CreateTask(
            app_name="HITL APP",
            title="Action Required: Review classification",
            data={
                "Question": "agent question",
            },
            app_version=1,
            assignee=None,
            app_folder_path="app-folder-path",
        )
    )

    # sleep 1 sec to increase the new trigger timestamp
    time.sleep(1)

    return Command(update={"message": response["Answer"]})


def wait_action_node(state: GraphState) -> Output:
    print("Response from HITL action: {}".format(state.message))

    response = interrupt(
        WaitTask(
            action=Task(
                key="1662478a-65b4-4a09-8e22-d707e5bd64f3",
                data={"Question": "agent question from wait action"},
            )
        )
    )
    return Output(message=response["Answer"])


builder = StateGraph(GraphState, input_schema=Input, output_schema=Output)

builder.add_node("create_action_node", create_action_node)
builder.add_node("wait_action_node", wait_action_node)

builder.add_edge(START, "create_action_node")
builder.add_edge("create_action_node", "wait_action_node")
builder.add_edge("wait_action_node", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
