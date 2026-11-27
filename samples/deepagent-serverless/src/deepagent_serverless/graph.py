"""DeepAgent that delegates filesystem operations and code execution to a serverless sandbox."""

from langchain_tavily import TavilySearch
from uipath_langchain.chat import UiPathChatOpenAI

from deepagents import create_deep_agent
from deepagent_serverless.serverless_backend import ServerlessBackend, ServerlessConfig

llm = UiPathChatOpenAI()
tavily_tool = TavilySearch(max_results=5)

MAIN_AGENT_PROMPT = """You are a coding assistant that can write and execute code on a remote serverless sandbox.

When given a task:
1. Delegate research to the research_specialist subagent if you need information
2. Write code files using the write_file tool (always under /tmp/workspace/)
3. Execute them using the execute tool (e.g. "python /tmp/workspace/script.py")
4. Use the execution output to verify results"""

RESEARCH_SUBAGENT_PROMPT = """You are a research specialist. Use web search to find relevant information.
Be thorough but concise. Cite your sources."""

research_subagent = {
    "name": "research_specialist",
    "description": "Specialized agent for gathering information using internet search",
    "system_prompt": RESEARCH_SUBAGENT_PROMPT,
    "tools": [tavily_tool],
    "model": llm,
}

deep_agent = create_deep_agent(
    model=llm,
    backend=ServerlessBackend(ServerlessConfig(sandbox_process_name="sandbox-deepagent")),
    system_prompt=MAIN_AGENT_PROMPT,
    tools=[tavily_tool],
    subagents=[research_subagent],
)
