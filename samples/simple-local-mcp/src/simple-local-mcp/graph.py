import sys
from contextlib import asynccontextmanager

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

model = ChatAnthropic(model="claude-3-7-sonnet-latest")


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
    agent = create_agent(model, await client.get_tools())
    yield agent
