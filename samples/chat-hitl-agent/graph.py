from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

tavily_tool = TavilySearch(max_results=5)

system_prompt = """
You are a Culinary Research & Recipe Assistant.

You help users:
- Research ingredients, cooking methods, and food science.
- Provide accurate recipes and substitutions.
- Recommend dishes based on preferences.
- Explain techniques clearly and safely.

Use TavilySearch whenever external information is useful.
Be concise, helpful, and food-savvy.
"""

llm = ChatAnthropic(model="claude-3-7-sonnet-latest")

graph = create_agent(
    model=llm,
    tools=[tavily_tool],
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "tavily_search": True  # Ask for approval before executing the search
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
)
