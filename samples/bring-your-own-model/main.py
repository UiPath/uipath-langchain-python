from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from uipath_langchain.chat import UiPathChatOpenAI
from pydantic import BaseModel


class GraphState(BaseModel):
    topic: str


class GraphOutput(BaseModel):
    report: str

async def generate_report(state: GraphState) -> GraphOutput:
    system_prompt = "You are a report generator. Please provide a brief report based on the given topic."
    llm = UiPathChatOpenAI(
        byo_connection_id="my-custom-model",
        model_name="gpt-4o-2024-11-20"
    )
    output = await llm.ainvoke(
        [SystemMessage(system_prompt), HumanMessage(state.topic)]
    )
    return GraphOutput(report=output.content)


builder = StateGraph(GraphState, output_schema=GraphOutput)

builder.add_node("generate_report", generate_report)

builder.add_edge(START, "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()
