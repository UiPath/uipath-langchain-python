"""Simple DeepAgent using Tavily search for research tasks."""

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import START, END, StateGraph, MessagesState
from deepagents import create_deep_agent

# Initialize tools
tavily_tool = TavilySearch(max_results=5)

# System prompt for the main agent
MAIN_AGENT_PROMPT = """You are a research assistant that helps users gather information and create reports.

When given a research topic:
1. Break down the research question into specific sub-questions if needed
2. Use the tavily_search tool to find relevant information directly
3. You can also delegate research tasks to specialized subagents:
   - research_specialist: for gathering specific information using internet search
   - quality_reviewer: for reviewing research outputs for accuracy and completeness
4. Organize your findings clearly and concisely
5. Provide your final answer with sources

Keep your responses focused and well-structured."""

# System prompt for the research subagent
RESEARCH_SUBAGENT_PROMPT = """You are a specialized research assistant focused on gathering specific information.

Your role:
1. Search for information using the internet_search tool
2. Extract the most relevant facts
3. Cite your sources
4. Return a clear, focused summary

Be thorough but concise in your research."""

# System prompt for the critique subagent
CRITIQUE_SUBAGENT_PROMPT = """You are a quality assurance specialist that reviews research outputs.

Your role:
1. Review the research findings for accuracy and completeness
2. Check if all key aspects of the question were addressed
3. Verify that sources are properly cited
4. Suggest improvements or gaps that need to be filled

Provide constructive feedback to improve the final output."""

# Initialize the LLM
llm = ChatAnthropic(model="claude-haiku-4-5")

# Define subagents as dictionaries
research_subagent = {
    "name": "research_specialist",
    "description": "Specialized agent for gathering specific information using internet search",
    "system_prompt": RESEARCH_SUBAGENT_PROMPT,
    "tools": [tavily_tool],
    "model": llm,
}

critique_subagent = {
    "name": "quality_reviewer",
    "description": "Quality assurance specialist that reviews research outputs for accuracy and completeness",
    "system_prompt": CRITIQUE_SUBAGENT_PROMPT,
    "tools": [],  # Critique agent doesn't need search tools
    "model": llm,
}

# Create main deep agent
deep_agent = create_deep_agent(
    model=llm,
    system_prompt=MAIN_AGENT_PROMPT,
    tools=[tavily_tool],
    subagents=[research_subagent, critique_subagent]
)

# Wrap the deep agent in a StateGraph with a simple schema for compatibility
async def agent_node(state: MessagesState) -> MessagesState:
    """Node that runs the deep agent."""
    result = await deep_agent.ainvoke(state)
    return result

# Build wrapper graph with standard MessagesState
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# Compile the wrapper graph
graph = builder.compile()
