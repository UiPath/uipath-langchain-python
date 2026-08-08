from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from uipath_langchain.chat import UiPathChat
from pydantic import BaseModel
from uipath.platform.common import InvokeSystemAgent
from langgraph.types import interrupt

llm = UiPathChat(model="gpt-4o-mini-2024-07-18")

class GraphInput(BaseModel):
    system_agent_name: str
    entrypoint: str
    system_agent_input: dict[str, Any] | None = None
    folder_path: str

class GraphOutput(BaseModel):
    sys_agent_response: str | None


async def generate_report(state: GraphInput) -> GraphOutput:
    system_agent_response = interrupt(InvokeSystemAgent(
        agent_name=state.system_agent_name,
        entrypoint=state.entrypoint,
        input_arguments=state.system_agent_input,
        folder_path=state.folder_path,
    ))
    return GraphOutput(sys_agent_response=str(system_agent_response))


builder = StateGraph(output_schema=GraphOutput, input_schema=GraphInput, state_schema=GraphInput)

builder.add_node("generate_report", generate_report)

builder.add_edge(START, "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()
