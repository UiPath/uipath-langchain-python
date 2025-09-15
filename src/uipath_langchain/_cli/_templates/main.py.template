from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from uipath_langchain.chat import UiPathChat
from pydantic import BaseModel

llm = UiPathChat(model="gpt-4o-mini-2024-07-18")


class GraphState(BaseModel):
    topic: str


class GraphOutput(BaseModel):
    report: str


async def generate_report(state: GraphState) -> GraphOutput:
    system_prompt = "You are a report generator. Please provide a brief report based on the given topic."
    output = await llm.ainvoke(
        [SystemMessage(system_prompt), HumanMessage(state.topic)]
    )
    return GraphOutput(report=output.content)


builder = StateGraph(GraphState, output=GraphOutput)

builder.add_node("generate_report", generate_report)

builder.add_edge(START, "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()
