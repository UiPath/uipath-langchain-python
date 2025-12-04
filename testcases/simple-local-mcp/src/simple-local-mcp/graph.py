import sys
import os
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from uipath_langchain.chat import UiPathAzureChatOpenAI, UiPathChat

model = UiPathChat(model="gpt-4o-2024-08-06", streaming=False)

@asynccontextmanager
async def make_graph():
    client = MultiServerMCPClient({
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
        })
    agent = create_agent(model, tools=await client.get_tools())
    yield agent
