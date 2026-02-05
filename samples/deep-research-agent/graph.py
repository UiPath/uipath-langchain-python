"""Deep Research Agent with Subagents."""

import json
import os
import sys
from pathlib import Path

# Add src directory to path for direct source imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from uipath_langchain.agent.deep import create_deep_agent
from uipath_langchain.chat import UiPathChat

# Search provider: "tavily" or "uipath" (default: uipath)
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "uipath").lower()

if SEARCH_PROVIDER == "tavily":
    from langchain_tavily import TavilySearch
    web_search = TavilySearch(max_results=5)
else:
    from uipath.agent.models.agent import AgentIntegrationToolResourceConfig
    from uipath_langchain.agent.tools import create_integration_tool

    # Load web search config from JSON
    config_path = Path(__file__).parent / "web_search_config.json"
    with open(config_path) as f:
        web_search_dict = json.load(f)

    # Convert dict to AgentIntegrationToolResourceConfig
    web_search_config = AgentIntegrationToolResourceConfig(**web_search_dict)
    web_search = create_integration_tool(web_search_config)

# Models
main_model = UiPathChat(model="gpt-4.1-2025-04-14", temperature=0)
fast_model = UiPathChat(model="gpt-4.1-mini-2025-04-14", temperature=0)

# Subagents
researcher = {
    "name": "researcher",
    "description": "Searches the web extensively for information. Use for gathering facts, finding sources, and exploring topics in depth.",
    "model": fast_model,
    "tools": [web_search],
    "system_prompt": """You are an expert web researcher specializing in deep, thorough research.

Your research approach:
1. MULTIPLE SEARCHES: Never rely on a single search. Run at least 3-5 different searches with varied queries:
   - Broad overview queries
   - Specific technical queries
   - Recent news/updates queries
   - Comparison/alternative queries
   - Expert opinion/discussion queries

2. EXPLORE THOROUGHLY: For each search result:
   - Extract key facts, statistics, and quotes
   - Note the source credibility and date
   - Identify technical details and specifications
   - Find real-world examples and case studies

3. CITE EVERYTHING: Every fact must have a source URL

4. ORGANIZE FINDINGS by category:
   - Core concepts and definitions
   - Technical details and specifications
   - Use cases and applications
   - Pros and cons
   - Recent developments
   - Expert opinions

Return comprehensive, well-organized findings. Quantity and depth matter.""",
}

reviewer = {
    "name": "reviewer",
    "description": "Reviews research for completeness, accuracy, and depth. Will request additional research if gaps are found.",
    "model": fast_model,
    "tools": [],
    "system_prompt": """You are a rigorous research quality reviewer. Your job is to ensure research is COMPREHENSIVE and DEEP.

Review criteria:
1. COMPLETENESS: Does the research cover ALL aspects of the topic?
   - Background and context
   - Technical details
   - Practical applications
   - Comparisons to alternatives
   - Recent developments
   - Future outlook

2. DEPTH: Is each aspect explored thoroughly?
   - Are there specific examples?
   - Are there statistics or data?
   - Are expert opinions included?
   - Are edge cases considered?

3. CITATIONS: Every major claim must have a source URL

4. GAPS: Identify specific missing information:
   - "Missing: comparison with [X]"
   - "Missing: recent updates from 2024-2025"
   - "Missing: performance benchmarks"
   - "Missing: real-world case studies"

If research is incomplete, LIST SPECIFIC GAPS that need more research.
Only approve when research is truly comprehensive.""",
}

# Main agent
graph = create_deep_agent(
    model=main_model,
    system_prompt="""You are a senior research lead producing comprehensive technical reports.

YOUR MISSION: Produce DEEP, THOROUGH research reports that would satisfy an expert audience.

TEAM:
- researcher: Conducts extensive web searches. Use multiple times for different aspects.
- reviewer: Quality control. Will identify gaps requiring more research.

RESEARCH PROCESS (follow this strictly):
1. PLAN: Break the topic into 5-7 key research questions/aspects
2. RESEARCH PHASE 1: Delegate each aspect to researcher (use researcher multiple times)
3. REVIEW: Have reviewer check for gaps
4. RESEARCH PHASE 2: Fill any gaps identified by reviewer
5. SYNTHESIZE: Compile into comprehensive report

IMPORTANT:
- Use the researcher subagent MULTIPLE TIMES for different aspects
- Don't accept shallow results - push for depth
- Include specific examples, statistics, and expert quotes
- Cover recent developments (2024-2025)
- NEVER ask the user clarifying questions or whether they want a report - ALWAYS proceed directly with the full research process and produce the complete report
- Do NOT ask for confirmation before generating the report - just do it

FINAL REPORT FORMAT:
## Executive Summary
[2-3 paragraph overview]

## Background & Context
[History, evolution, why this matters]

## Core Concepts
[Detailed technical explanation]

## Key Features & Capabilities
[Comprehensive list with details]

## Real-World Applications
[Specific use cases and case studies]

## Comparisons & Alternatives
[How it compares to alternatives]

## Pros & Cons
[Balanced analysis]

## Recent Developments
[Updates from 2024-2025]

## Future Outlook
[Trends and predictions]

## Sources
[All URLs cited]

The report should be 1500-3000 words. Quality and depth are paramount.""",
    tools=[web_search],
    subagents=[researcher, reviewer],
)
