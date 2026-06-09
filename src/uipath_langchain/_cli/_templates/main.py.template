from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from uipath_langchain.chat import (
    UiPathAzureChatOpenAI,
    UiPathChatAnthropicBedrock,
    UiPathChatGoogleGenerativeAI,
)
from pydantic import BaseModel


class GraphState(BaseModel):
    topic: str


class GraphOutput(BaseModel):
    report: str


async def generate_report(state: GraphState) -> GraphOutput:
    # Choose your LLM provider by uncommenting one of the following:
    llm = UiPathChatAnthropicBedrock(model="anthropic.claude-haiku-4-5-20251001-v1:0")
    # llm = UiPathAzureChatOpenAI(model="gpt-4.1-mini-2025-04-14")
    # llm = UiPathChatGoogleGenerativeAI(model="gemini-2.5-flash")
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
