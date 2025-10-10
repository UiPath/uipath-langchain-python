from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic.dataclasses import dataclass
from uipath.models import InvokeProcess
from uipath.tracing import traced

@dataclass
class BindingAgentInput:
    pass

@dataclass
class BindingAgentOutput:
    pass

@traced(name="run_process")
async def run_process(*args, **kwargs) -> {}:
    print(f"input args: {args}, kwargs: {kwargs}")
    result = interrupt(
        InvokeProcess(
            name="A2ALoanCreditRatingTool",
            input_arguments={"name": "John Doe"}
        ),
    )
    print(f"result: {result}")
    return {}

builder = StateGraph(state_schema=BindingAgentInput, input=BindingAgentInput, output=BindingAgentOutput)

builder.add_node("run_process", run_process)
builder.add_edge(START, "run_process")
builder.add_edge("run_process", END)

graph = builder.compile()
