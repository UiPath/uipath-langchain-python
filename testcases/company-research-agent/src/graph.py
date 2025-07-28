import os
from typing import Union

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from uipath_langchain.chat import UiPathAzureChatOpenAI, UiPathChat 

# Configuration constants
MAX_SEARCH_RESULTS = 5
DEFAULT_MODEL = "gpt-4o-2024-08-06"
ALTERNATIVE_MODEL = "claude-3-5-sonnet-latest"


def get_search_tool() -> Union[TavilySearchResults, DuckDuckGoSearchResults]:
    """Get the appropriate search tool based on available API keys."""
    if os.getenv("TAVILY_API_KEY"):
        return TavilySearchResults(max_results=MAX_SEARCH_RESULTS)
    return DuckDuckGoSearchResults()


# System prompt for the research agent
SYSTEM_PROMPT = """You are an advanced AI assistant specializing in corporate research and outreach strategy development. Your primary functions are:
1. Researching target companies: Gather comprehensive information about the specified company, including its history, industry position, products/services, and recent news.
2. Analyzing organizational structures: Investigate and outline the company's internal structure, departments, and hierarchy.
3. Identifying key decision-makers: Find and provide information on important executives and leaders within the company who are likely to be involved in decision-making processes.
4. Developing outreach strategies: Based on the gathered information, create tailored strategies for effective communication and engagement with the company and its key personnel.

To accomplish these tasks, follow these steps:
1. Use the TavilySearchResults tool to find recent and relevant information about the company.
2. Analyze the collected data to form insights about the company's structure, key decision-makers, and potential outreach strategies.

When using the search tool:
- Clearly state the purpose of each search.
- Formulate effective search queries to find specific information about different aspects of the company.
- If a search doesn't provide the expected information, try refining your query.

When responding, structure your output as a comprehensive analysis. Use clear section headers to organize the information. Provide concise, actionable insights. If you need more information to complete any part of your analysis, clearly state what additional details would be helpful.

Always maintain a professional and objective tone in your research and recommendations. Your goal is to provide accurate, valuable information that can be used to inform business decisions and outreach efforts.

DO NOT do any math as specified in your instructions.
"""


def create_llm() -> Union[UiPathAzureChatOpenAI, UiPathChat]:
    """Create and configure the language model based on an environment variable."""
    if os.getenv("USE_AZURE_CHAT", "false").lower() == "true":
        return UiPathAzureChatOpenAI(model=DEFAULT_MODEL)
    return UiPathChat(model=DEFAULT_MODEL)


def create_research_agent():
    """Create the research agent with configured LLM and tools."""
    llm = create_llm()
    search_tool = get_search_tool()
    return create_react_agent(llm, tools=[search_tool], prompt=SYSTEM_PROMPT)


class GraphState(BaseModel):
    """State model for the research graph."""
    company_name: str


class GraphOutput(BaseModel):
    """Output model for the research graph."""
    response: str


def create_user_message(company_name: str) -> str:
    """Create a formatted user message for company research."""
    return f"""Please provide a comprehensive analysis and outreach strategy for the company: {company_name}. Use the TavilySearchResults tool to gather information. Include detailed research on the company's background, organizational structure, key decision-makers, and a tailored outreach strategy. Format your response using the following section headers:

1. Company Overview
2. Organizational Structure
3. Key Decision-Makers
4. Outreach Strategy
5. Additional Information Needed (if applicable)

Ensure that each section is clearly labeled and contains relevant, concise information. If you need any additional specific information about {company_name} or its industry to enhance your analysis, please include your questions in the 'Additional Information Needed' section.
"""


async def research_node(state: GraphState) -> GraphOutput:
    """Research node that performs company analysis."""
    research_agent = create_research_agent()
    user_message = create_user_message(state.company_name)
    
    # Create message state for the agent
    message_state = MessagesState(messages=[{"role": "user", "content": user_message}])
    
    # Invoke the research agent
    result = await research_agent.ainvoke(message_state)
    
    return GraphOutput(response=result["messages"][-1].content)


def build_research_graph() -> StateGraph:
    """Build and compile the research graph."""
    builder = StateGraph(GraphState, output=GraphOutput)
    
    # Add nodes
    builder.add_node("researcher", research_node)
    
    # Add edges
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", END)
    
    return builder.compile()


# Create the compiled graph
graph = build_research_graph()
