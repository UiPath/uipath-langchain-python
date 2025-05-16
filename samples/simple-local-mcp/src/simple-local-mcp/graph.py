import sys
from contextlib import asynccontextmanager

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from uipath_langchain.chat.models import UiPathChat
import os

if os.getenv("USE_UIPATH_AI_UNITS") and os.getenv("USE_UIPATH_AI_UNITS") == "true":
    # other available UiPath chat models
    # "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # "anthropic.claude-3-7-sonnet-20250219-v1:0",
    # "anthropic.claude-3-haiku-20240307-v1:0",
    # "gemini-1.5-pro-001",
    # "gemini-2.0-flash-001",
    # "gpt-4o-2024-05-13",
    # "gpt-4o-2024-08-06",
    # "gpt-4o-2024-11-20",
    # "gpt-4o-mini-2024-07-18",
    # "o3-mini-2025-01-31",
    llm = UiPathChat(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
else:
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

@asynccontextmanager
async def make_graph():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": ["src/simple-local-mcp/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "command": sys.executable,
                "args": ["src/simple-local-mcp/weather_server.py"],
                "transport": "stdio",
            },
        }
    ) as client:
        agent = create_react_agent(llm, client.get_tools())
        yield agent
