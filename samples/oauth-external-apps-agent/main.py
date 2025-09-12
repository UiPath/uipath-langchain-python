import dotenv
import os
from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from uipath import UiPath
from typing import Optional, Literal
from mcp.client.streamable_http import streamablehttp_client
import httpx
from langchain.schema import SystemMessage, HumanMessage
from dotenv import set_key, find_dotenv
dotenv.load_dotenv()

class GraphInput(BaseModel):
    """Input for our OAuth flow graph."""
    task: str

class GraphOutput(BaseModel):
    """Output for our OAuth flow graph."""
    result: str

class State(BaseModel):
    """State for our LangGraph nodes."""
    task: str
    access_token: Optional[str] = os.getenv("UIPATH_ACCESS_TOKEN")
    response: Optional[str] = None

async def fetch_new_access_token(state: State) -> Command:
    """Fetches a new OAuth token if the current one is expired or missing."""
    try:
        uipath_client = UiPath(
            base_url=os.getenv("UIPATH_TOKEN_URL"),
            client_id=os.getenv("UIPATH_CLIENT_ID"),
            client_secret=os.getenv("UIPATH_CLIENT_SECRET"),
            scope=os.getenv("SCOPE"),
        )
        # asset = uipath_client.assets.retrieve(name="test-asset", folder_path="TestFolder")
        # print(f"Asset: {asset}")

        new_access_token = uipath_client._config.secret

        os.environ["UIPATH_ACCESS_TOKEN"] = new_access_token
        env_path = find_dotenv(usecwd=True)
        if env_path:
            set_key(env_path, "UIPATH_ACCESS_TOKEN", new_access_token, quote_mode="never")

        return Command(update={"access_token": new_access_token})

    except Exception as e:
        raise Exception(f"Failed to fetch and update the new access token: {str(e)}")

@asynccontextmanager
async def agent_mcp(access_token: str):
    async with streamablehttp_client(
        url=os.getenv("UIPATH_MCP_SERVER_URL"),
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=60,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            model = ChatAnthropic(model="claude-3-5-sonnet-latest")
            agent = create_react_agent(model, tools=tools)
            yield agent

async def connect_to_mcp(state: State) -> Command:
    """Attempts to establish MCP connectivity using the access token."""
    try:
        async with agent_mcp(state.access_token) as agent:
            system_context = SystemMessage(
                content="You are connected to the MCP system. Execute the following task."
            )
            human_task = HumanMessage(content=state.task)

            agent_response = await agent.ainvoke({
                "messages": [system_context, human_task],
            })
            results = agent_response["messages"][-1].content
            return Command(update={"response": results})
    except ExceptionGroup as e:
        for error in e.exceptions:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 401:
                return Command(update={"access_token": None, "response": "Token expired. Needs refresh."})
        raise e

def decide_next_node(state: State) -> Literal["fetch_new_access_token", "connect_to_mcp", "collect_output"]:
    """Decides whether to fetch a new token or proceed to the final output."""
    if state.access_token is None:
        return "fetch_new_access_token"
    return "collect_output"

def collect_output(state: State) -> GraphOutput:
    """Collects the output after the processing is complete."""
    return GraphOutput(result=state.response)

builder = StateGraph(State, input=GraphInput, output=GraphOutput)

builder.add_node("fetch_new_access_token", fetch_new_access_token)
builder.add_node("connect_to_mcp", connect_to_mcp)
builder.add_node("collect_output", collect_output)

builder.add_edge(START, "connect_to_mcp")
builder.add_conditional_edges(
    "connect_to_mcp",
    decide_next_node,
)
builder.add_edge("fetch_new_access_token", "connect_to_mcp")
builder.add_edge("collect_output", END)

graph = builder.compile()
