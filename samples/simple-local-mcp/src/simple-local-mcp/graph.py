import sys
from contextlib import AsyncExitStack, asynccontextmanager

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from uipath_langchain.agent.tools.mcp import load_mcp_tools

model = ChatAnthropic(model="claude-3-7-sonnet-latest")


@asynccontextmanager
async def make_graph():
    async with AsyncExitStack() as stack:
        tools = []
        for script in ("math_server.py", "weather_server.py"):
            read, write = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[f"src/simple-local-mcp/{script}"],
                    )
                )
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools.extend(await load_mcp_tools(session))

        agent = create_agent(model, tools=tools)
        yield agent
