from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from uipath_langchain.chat import hitl_tool

tavily_tool = TavilySearch(max_results=5)


@hitl_tool
def search_web(query: str) -> str:
    """Search the web for information using Tavily."""
    return tavily_tool.invoke({"query": query})

system_prompt = """
You are a Culinary Research & Recipe Assistant.

You help users:
- Research ingredients, cooking methods, and food science.
- Provide accurate recipes and substitutions.
- Recommend dishes based on preferences.
- Explain techniques clearly and safely.

Use search_web whenever external information is useful.
Be concise, helpful, and food-savvy.
"""

llm = ChatAnthropic(model="claude-3-7-sonnet-latest")

graph = create_agent(
    model=llm,
    tools=[search_web],
    system_prompt=system_prompt,
)
