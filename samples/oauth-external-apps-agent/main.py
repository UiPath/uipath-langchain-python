import os
import dotenv
import httpx
from contextlib import asynccontextmanager
from typing import Optional, Literal

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

from uipath_langchain.chat.models import UiPathChat
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from uipath.platform import UiPath

dotenv.load_dotenv()

UIPATH_CLIENT_ID = "EXTERNAL_APP_CLIENT_ID_HERE"
UIPATH_CLIENT_SECRET = os.getenv("UIPATH_CLIENT_SECRET")
UIPATH_SCOPE = "OR.Jobs"
UIPATH_URL = "base_url"
UIPATH_MCP_SERVER_URL = os.getenv("UIPATH_MCP_SERVER_URL")

class GraphInput(BaseModel):
    task: str

class GraphOutput(BaseModel):
    result: str

class State(BaseModel):
    task: str
    access_token: Optional[str] = os.getenv("UIPATH_ACCESS_TOKEN")
    result: Optional[str] = None

async def fetch_new_access_token(state: State) -> Command:
    try:
        UiPath(
            base_url=UIPATH_URL,
            client_id=UIPATH_CLIENT_ID,
            client_secret=UIPATH_CLIENT_SECRET,
            scope=UIPATH_SCOPE,
        )
        return Command(update={"access_token": os.getenv("UIPATH_ACCESS_TOKEN")})

    except Exception as e:
        raise Exception(f"Failed to initialize UiPath SDK: {str(e)}")

@asynccontextmanager
async def agent_mcp(access_token: str):
    async with streamablehttp_client(
        url=UIPATH_MCP_SERVER_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=60,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            model = UiPathChat(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
            agent = create_agent(model, tools=tools)
            yield agent

async def connect_to_mcp(state: State) -> Command:
    try:
        async with agent_mcp(state.access_token) as agent:
            agent_response = await agent.ainvoke({
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=state.task),
                ],
            })
            return Command(update={"result": agent_response["messages"][-1].content})
    except ExceptionGroup as e:
        for error in e.exceptions:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 401:
                return Command(update={"access_token": None})
        raise

def route_start(state: State) -> Literal["fetch_new_access_token", "connect_to_mcp"]:
    return "fetch_new_access_token" if state.access_token is None else "connect_to_mcp"

def route_after_connect(state: State):
    return "fetch_new_access_token" if state.access_token is None else END

builder = StateGraph(State, input=GraphInput, output=GraphOutput)
builder.add_node("fetch_new_access_token", fetch_new_access_token)
builder.add_node("connect_to_mcp", connect_to_mcp)

builder.add_conditional_edges(START, route_start)
builder.add_edge("fetch_new_access_token", "connect_to_mcp")
builder.add_conditional_edges("connect_to_mcp", route_after_connect)

graph = builder.compile()
