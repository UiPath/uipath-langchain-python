import asyncio
from typing import Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph
from uipath._cli.middlewares import MiddlewareResult
from uipath_langchain._cli.cli_run import (
    LangGraphRuntime,
    LangGraphRuntimeContext,
)

load_dotenv()


def lowcode_run_middleware(
    entrypoint: Optional[str], input: Optional[str], resume: bool, **kwargs
) -> MiddlewareResult:
    # read agent.json file
    # parse it into a python class
    # create a LangGraph:StateGraph based on the agent definition
    agent_graph = StateGraph()

    async def execute():
        context: LangGraphRuntimeContext = LangGraphRuntimeContext()
        context.entrypoint = entrypoint
        context.input = input
        context.resume = resume
        context.kwargs = kwargs

        context.state_graph = agent_graph

        async with LangGraphRuntime.from_context(context) as runtime:
            await runtime.execute()

    return asyncio.run(execute())
