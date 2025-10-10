from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from uipath_langchain.chat import UiPathChat
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()


# Initialize LLM
llm = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError(
        "TAVILY_API_KEY environment variable is not set. "
        "Please set it in your .env file or environment. "
        "Get your API key from https://tavily.com"
    )
tavily_client = TavilyClient(api_key=tavily_api_key)


# Define tools for the research agent
@tool
def web_search(query: str) -> str:
    """Search the web for current information about a company, industry, or person.
    Use this tool to gather up-to-date information about companies, their leadership,
    products, recent news, and market position. Returns real-time search results."""
    try:
        # Perform Tavily search
        response = tavily_client.search(
            query=query,
            search_depth="advanced",  # Use advanced search for more comprehensive results
            max_results=5,
            include_answer=True,  # Get AI-generated answer summary
            include_raw_content=False,  # Don't need raw HTML
        )

        # Format results
        results = []

        # Add the AI-generated answer if available
        if response.get("answer"):
            results.append(f"Summary: {response['answer']}\n")

        # Add individual search results
        for idx, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "")

            results.append(
                f"{idx}. {title}\n"
                f"   {content}\n"
                f"   Source: {url}\n"
            )

        return "\n".join(results) if results else "No search results found."

    except Exception as e:
        return f"Error performing web search: {str(e)}"


# Input/Output Models
class Input(BaseModel):
    company_name: str = Field(description="The name of the company to research")


class PersonnelInfo(BaseModel):
    name: str = Field(description="Full name of the key person")
    role: str = Field(description="Job title or role in the company")
    background: str = Field(description="Brief background and relevant experience")
    communication_style: Optional[str] = Field(default=None, description="Recommended communication approach")


class EngagementStrategy(BaseModel):
    strategy_name: str = Field(description="Name of the engagement strategy")
    description: str = Field(description="Detailed description of the strategy")
    target_audience: str = Field(description="Who this strategy targets")
    key_points: List[str] = Field(description="Key talking points or actions")
    timing_recommendations: str = Field(description="When and how to implement")


class Output(BaseModel):
    company_overview: str = Field(description="Comprehensive overview of the company")
    industry_context: str = Field(description="Industry position and competitive landscape")
    key_personnel: List[PersonnelInfo] = Field(description="Key decision makers and influencers")
    engagement_strategies: List[EngagementStrategy] = Field(description="Tailored engagement strategies")
    communication_recommendations: str = Field(description="Overall communication approach")
    next_steps: List[str] = Field(description="Recommended next steps for engagement")


# State Model
class State(BaseModel):
    company_name: str
    research_data: str = ""
    company_analysis: str = ""
    personnel_data: List[PersonnelInfo] = []
    strategies: List[EngagementStrategy] = []
    final_recommendations: str = ""


# Node 1: Research Company
async def research_company(state: State) -> State:
    """Conduct comprehensive research on the company using available tools."""

    # Create a research agent with tools
    research_agent = create_react_agent(
        llm,
        tools=[web_search],
        prompt="""You are a professional business intelligence researcher. Your task is to gather
comprehensive information about companies. Use the web_search tool to find:
- Company background, history, and mission
- Products and services offered
- Market position and competitors
- Recent news and developments
- Financial performance and growth trajectory
- Company culture and values
- Key leadership and executives

Be thorough and factual. Gather as much relevant information as possible."""
    )

    research_query = f"""Research the company '{state.company_name}'.
Gather comprehensive information including:
1. Company overview and history
2. Products/services and market position
3. Recent news and developments
4. Leadership team and key personnel
5. Company culture and values
6. Financial performance and growth
7. Industry context and competitors

Provide detailed, factual information."""

    result = await research_agent.ainvoke({"messages": [("user", research_query)]})
    research_data = result["messages"][-1].content

    return State(
        company_name=state.company_name,
        research_data=research_data
    )


# Node 2: Analyze Company
async def analyze_company(state: State) -> State:
    """Analyze the research data to create a structured company analysis."""

    analysis_prompt = f"""Based on the following research about {state.company_name}, create a comprehensive analysis:

Research Data:
{state.research_data}

Provide:
1. Company Overview: Core business, mission, and market position
2. Industry Context: Competitive landscape and market trends
3. Strengths and Opportunities: What makes them unique and potential areas for collaboration
4. Communication Considerations: Company culture, values, and preferred engagement styles

Be analytical and strategic in your assessment."""

    response = await llm.ainvoke([HumanMessage(analysis_prompt)])

    return State(
        company_name=state.company_name,
        research_data=state.research_data,
        company_analysis=response.content
    )


# Node 3: Identify Key Personnel
async def identify_personnel(state: State) -> State:
    """Extract and analyze key personnel from the research data."""

    # Use structured output for personnel extraction
    class PersonnelExtraction(BaseModel):
        personnel: List[PersonnelInfo] = Field(description="List of key personnel identified")

    structured_llm = llm.with_structured_output(PersonnelExtraction)

    personnel_prompt = f"""Based on the research data about {state.company_name}, identify key personnel:

Research Data:
{state.research_data}

Company Analysis:
{state.company_analysis}

Identify 3-7 key personnel including:
- C-level executives (CEO, CFO, CTO, etc.)
- Department heads relevant to potential partnerships
- Innovation or strategy leaders
- Public-facing representatives

For each person, provide:
- Full name and role
- Brief background and relevant experience
- Recommended communication style based on their role and public information"""

    personnel_result = await structured_llm.ainvoke(personnel_prompt)

    # Handle both dict and object responses
    if isinstance(personnel_result, dict):
        personnel_list = personnel_result.get('personnel', [])
    else:
        personnel_list = personnel_result.personnel

    return State(
        company_name=state.company_name,
        research_data=state.research_data,
        company_analysis=state.company_analysis,
        personnel_data=personnel_list
    )


# Node 4: Generate Engagement Strategies
async def generate_strategies(state: State) -> State:
    """Create tailored engagement and communication strategies."""

    class StrategyGeneration(BaseModel):
        strategies: List[EngagementStrategy] = Field(description="List of tailored engagement strategies")

    structured_llm = llm.with_structured_output(StrategyGeneration)

    strategy_prompt = f"""Based on the comprehensive research about {state.company_name}, create tailored engagement strategies:

Company Analysis:
{state.company_analysis}

Key Personnel:
{[f"- {p.name} ({p.role}): {p.background}" for p in state.personnel_data]}

Create 4-6 specific engagement strategies that include:
1. Executive Engagement Strategy: How to approach C-level executives
2. Technical/Product Strategy: Engaging with technical decision-makers
3. Partnership Development Strategy: Building long-term relationships
4. Value Proposition Strategy: Highlighting relevant benefits
5. Communication Channel Strategy: Best channels and formats
6. Follow-up and Nurture Strategy: Maintaining engagement

For each strategy, provide:
- Clear strategy name
- Detailed description
- Target audience within the company
- Key talking points or actions (3-5 points)
- Timing and implementation recommendations

Make strategies specific to this company's culture, industry, and current situation."""

    strategy_result = await structured_llm.ainvoke(strategy_prompt)

    # Handle both dict and object responses
    if isinstance(strategy_result, dict):
        strategies_list = strategy_result.get('strategies', [])
    else:
        strategies_list = strategy_result.strategies

    return State(
        company_name=state.company_name,
        research_data=state.research_data,
        company_analysis=state.company_analysis,
        personnel_data=state.personnel_data,
        strategies=strategies_list
    )


# Node 5: Create Final Recommendations
async def create_recommendations(state: State) -> State:
    """Generate overall communication recommendations and next steps."""

    recommendations_prompt = f"""Based on all the research and strategies for {state.company_name},
create a comprehensive communication recommendation document:

Company Analysis:
{state.company_analysis}

Key Personnel: {len(state.personnel_data)} identified
Strategies Developed: {len(state.strategies)}

Provide:
1. Overall Communication Approach: High-level guidance for engaging with this company
2. Key Success Factors: What will make engagement successful
3. Potential Pitfalls: What to avoid
4. Timeline Recommendations: Suggested sequence and timing of engagement
5. Resource Requirements: What you'll need to execute effectively

Be strategic, practical, and actionable."""

    response = await llm.ainvoke([HumanMessage(recommendations_prompt)])

    return State(
        company_name=state.company_name,
        research_data=state.research_data,
        company_analysis=state.company_analysis,
        personnel_data=state.personnel_data,
        strategies=state.strategies,
        final_recommendations=response.content
    )


# Node 6: Format Output
async def create_output(state: State) -> Output:
    """Format the final output with all research and strategies."""

    # Extract company overview and industry context from analysis
    company_sections = state.company_analysis.split('\n\n')
    company_overview = company_sections[0] if company_sections else state.company_analysis[:500]
    industry_context = company_sections[1] if len(company_sections) > 1 else "See company analysis for details."

    # Generate next steps
    next_steps = [
        f"Review the {len(state.personnel_data)} key personnel profiles and prioritize outreach targets",
        f"Select 2-3 engagement strategies from the {len(state.strategies)} options provided that best fit your goals",
        "Customize communication templates based on the recommended approaches",
        "Schedule initial outreach within the timeframes suggested",
        "Prepare relevant case studies and materials mentioned in the strategies",
        "Set up tracking system for engagement metrics and follow-ups"
    ]

    return Output(
        company_overview=company_overview,
        industry_context=industry_context,
        key_personnel=state.personnel_data,
        engagement_strategies=state.strategies,
        communication_recommendations=state.final_recommendations,
        next_steps=next_steps
    )


# Build the graph
builder = StateGraph(State, input=Input, output=Output)

# Add nodes
builder.add_node("research", research_company)
builder.add_node("analyze", analyze_company)
builder.add_node("identify_personnel", identify_personnel)
builder.add_node("generate_strategies", generate_strategies)
builder.add_node("create_recommendations", create_recommendations)
builder.add_node("output", create_output)

# Add edges for sequential processing
builder.add_edge(START, "research")
builder.add_edge("research", "analyze")
builder.add_edge("analyze", "identify_personnel")
builder.add_edge("identify_personnel", "generate_strategies")
builder.add_edge("generate_strategies", "create_recommendations")
builder.add_edge("create_recommendations", "output")
builder.add_edge("output", END)

# Compile the graph
graph = builder.compile()
