from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain.agents import create_agent
from pydantic import BaseModel

tavily_tool = TavilySearch(max_results=5)

llm = ChatAnthropic(model="claude-3-7-sonnet-latest")


research_agent = create_agent(
    llm, tools=[tavily_tool], system_prompt="You are a researcher. DO NOT do any math."
)


class GraphOutput(BaseModel):
    answer: str


async def research_node(state: MessagesState) -> GraphOutput:
    result = await research_agent.ainvoke(state)
    return GraphOutput(answer=result["messages"][-1].content)


# Build the state graph
builder = StateGraph(MessagesState, output=GraphOutput)
builder.add_node("researcher", research_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", END)

# Compile the graph
graph = builder.compile()
