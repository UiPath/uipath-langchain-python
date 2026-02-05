"""Deep Research Agent with Subagents."""

import json
import sys
from pathlib import Path

# Add src directory to path for direct source imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from uipath.agent.models.agent import AgentIntegrationToolResourceConfig
from uipath_langchain.agent.deep import create_deep_agent
from uipath_langchain.agent.tools import create_integration_tool
from uipath_langchain.chat import UiPathChat

# Load web search config from JSON
config_path = Path(__file__).parent / "web_search_config.json"
with open(config_path) as f:
    web_search_dict = json.load(f)

# Convert dict to AgentIntegrationToolResourceConfig
web_search_config = AgentIntegrationToolResourceConfig(**web_search_dict)

# Tools
web_search = create_integration_tool(web_search_config)

# Models
main_model = UiPathChat(model="gpt-4.1-2025-04-14", temperature=0)
fast_model = UiPathChat(model="gpt-4.1-mini-2025-04-14", temperature=0)

# Subagents
researcher = {
    "name": "researcher",
    "description": "Searches the web for information on technical topics.",
    "model": fast_model,
    "tools": [web_search],
    "system_prompt": """You are a web researcher. Find accurate, current information.

When researching:
1. Search for relevant sources
2. Extract key facts and examples
3. Cite sources with URLs

Return organized findings with citations.""",
}

reviewer = {
    "name": "reviewer",
    "description": "Reviews research for completeness and accuracy.",
    "model": fast_model,
    "tools": [],
    "system_prompt": """You are a research reviewer. Ensure quality.

When reviewing:
1. Check if findings answer the question
2. Identify gaps or missing aspects
3. Verify claims have citations

Be constructive and specific.""",
}

# Main agent
graph = create_deep_agent(
    model=main_model,
    system_prompt="""You are a research lead coordinating technical research.

Team:
- researcher: Web search for docs, tutorials, discussions
- reviewer: Quality check before final delivery

Process:
1. Break topic into questions
2. Delegate searches to researcher
3. Have reviewer verify findings
4. Synthesize into a structured report

Report format:
- Summary
- Key features
- Code examples (if relevant)
- Trade-offs
- Sources""",
    tools=[web_search],
    subagents=[researcher, reviewer],
)
