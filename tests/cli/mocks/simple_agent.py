from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


class GraphState(BaseModel):
    topic: str


class GraphOutput(BaseModel):
    report: str


async def generate_report(state: GraphState) -> GraphOutput:
    return GraphOutput(report=f" This is mock report for {state.topic}")


builder = StateGraph(GraphState, output_schema=GraphOutput)

builder.add_node("generate_report", generate_report)

builder.add_edge(START, "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()
