import sys
import os
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from uipath_langchain.chat import UiPathAzureChatOpenAI, UiPathChat

if os.getenv("USE_AZURE_CHAT", "false").lower() == "true":
    model = UiPathAzureChatOpenAI(model="gpt-4o-2024-08-06")
else:
    model = UiPathChat(model="gpt-4o-2024-08-06")

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
    agent = create_react_agent(model, await client.get_tools())
    yield agent
