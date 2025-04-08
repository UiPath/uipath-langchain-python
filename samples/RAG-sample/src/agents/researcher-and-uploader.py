from typing import Optional
import time
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from uipath import UiPath
from langchain_core.messages import HumanMessage, AIMessage


tavily_tool = TavilySearchResults(max_results=5)

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

uipath = UiPath()
research_agent = create_react_agent(
    llm, tools=[tavily_tool], prompt="You are a researcher. Search relevant information given the user topic. Don't do summarizations. Retrieve raw data."
)

class GraphInput(MessagesState):
    bucket_name: str
    bucket_folder: Optional[str]

class GraphState(MessagesState):
    web_results: str
    bucket_name: str
    bucket_folder: Optional[str]

def prepare_input(state: GraphInput) -> GraphState:
    return GraphState(
        messages=state["messages"],
        web_results="",
        bucket_name=state["bucket_name"],
        bucket_folder=state.get("bucket_folder",None),
    )

def research_node(state: GraphState) -> Command:
    result = research_agent.invoke(state)
    web_results = result["messages"][-1].content
    return Command(
        update={
            "web_results": web_results,
        })

def upload_to_bucket(state: GraphState) -> MessagesState:
    #TODO: also include a question summarization as document name
    current_timestamp = int(time.time())
    uipath.buckets.upload_from_memory(
        bucket_name=state["bucket_name"],
        blob_file_path=f"{current_timestamp}.txt",
        content_type="application/txt",
        content=state["web_results"],)
    return MessagesState(messages=[AIMessage("Relevant information uploaded to bucket.")])


# Build the state graph
builder = StateGraph(input=GraphInput, output=MessagesState)
builder.add_node("researcher", research_node)
builder.add_node("upload_to_bucket", upload_to_bucket)
builder.add_node("prepare_input", prepare_input)

builder.add_edge(START, "prepare_input")
builder.add_edge("prepare_input", "researcher")
builder.add_edge("researcher", "upload_to_bucket")
builder.add_edge("upload_to_bucket", END)

# Compile the graph
graph = builder.compile()
