import os
from typing import Any
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def mcp_client(state: MessagesState) -> dict[str, Any]:
    """Agent node that connects to MCP server and processes messages."""
    async with streamablehttp_client(
        url=os.getenv("UIPATH_MCP_SERVER_URL"),
        headers={"Authorization": f"Bearer {os.getenv('UIPATH_ACCESS_TOKEN')}"},
        timeout=60,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(f"Loaded {len(tools)} tools from MCP server")
            model = ChatAnthropic(model="claude-3-7-sonnet-latest")
            agent = create_agent(model, tools=tools)
            result = await agent.ainvoke(state)
            return result


builder = StateGraph(MessagesState)
builder.add_node("mcp_client", mcp_client)
builder.add_edge(START, "mcp_client")
builder.add_edge("mcp_client", END)

graph = builder.compile()
