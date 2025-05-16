from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from uipath_langchain.chat.models import UiPathChat
import os
from pydantic import BaseModel

tavily_tool = TavilySearchResults(max_results=5)

if os.getenv("USE_UIPATH_AI_UNITS") and os.getenv("USE_UIPATH_AI_UNITS") == "true":
    # other available UiPath chat models
    # "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # "anthropic.claude-3-7-sonnet-20250219-v1:0",
    # "anthropic.claude-3-haiku-20240307-v1:0",
    # "gemini-1.5-pro-001",
    # "gemini-2.0-flash-001",
    # "gpt-4o-2024-05-13",
    # "gpt-4o-2024-08-06",
    # "gpt-4o-2024-11-20",
    # "gpt-4o-mini-2024-07-18",
    # "o3-mini-2025-01-31",
    llm = UiPathChat(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
else:
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

research_agent = create_react_agent(
    llm, tools=[tavily_tool], prompt="You are a researcher. DO NOT do any math."
)


class GraphOutput(BaseModel):
    answer: str


async def research_node(state: MessagesState) -> GraphOutput:
    result = await research_agent.ainvoke(state)
    return GraphOutput(answer=result["messages"][-1].content)


# Build the state graph
builder = StateGraph(input=MessagesState, output=GraphOutput)
builder.add_node("researcher", research_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", END)

# Compile the graph
graph = builder.compile()
