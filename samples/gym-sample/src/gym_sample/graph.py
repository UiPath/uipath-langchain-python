import os
from typing import Any, AsyncGenerator, Dict, List
from contextlib import asynccontextmanager

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from gym_sample.uipath_gym_types import (
    AgentBaseClass,
    EndExecutionTool,
    StructuredTool,
    BasicLoop,
    Datapoint,
)
from uipath_langchain.chat import UiPathAzureChatOpenAI
from gym_sample.evals import get_datapoints


def get_model() -> BaseChatModel:
    """Get the ChatAnthropic model (created lazily to allow environment loading)."""
    return UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")


def get_agents() -> Dict[str, AgentBaseClass]:
    """Get the agents (created lazily to allow environment loading)."""
    return {
        "calculator":
            AgentBaseClass(
                system_prompt="You are a calculator agent. You can perform mathematical operations using the available tools. When you have completed the calculation, use the end_execution tool to provide your final result with a score (0.0 to 1.0 representing confidence) and observations about the calculation process.",
                user_prompt="Calculate the result of: {expression}.",
                tools=[
                    StructuredTool.from_function(
                        func=lambda a, b: a + b,
                        name="add",
                        description="Add two numbers",
                        args_schema={
                            "type": "object",
                            "properties": {
                                "a": {"type": "number", "description": "First number"},
                                "b": {"type": "number", "description": "Second number"}
                            },
                            "required": ["a", "b"]
                        },
                    ),
                    StructuredTool.from_function(
                        func=lambda a, b: a * b,
                        name="multiply",
                        description="Multiply two numbers",
                        args_schema={
                            "type": "object",
                            "properties": {
                                "a": {"type": "number", "description": "First number"},
                                "b": {"type": "number", "description": "Second number"}
                            },
                            "required": ["a", "b"]
                        },
                    ),
                ],
                end_execution_tool=EndExecutionTool(
                    args_schema={
                        "type": "object",
                        "properties": {
                            "answer": {"type": "number", "description": "The numerical answer to the calculation"}
                        },
                        "required": ["answer"]
                    }
                ),
                datapoints=get_datapoints(),
            ),
    }


@asynccontextmanager
async def agent(agent_name: str = "calculator", agent_input: Dict[str, Any] | int | None = None, debug: bool = False) -> AsyncGenerator[StateGraph, None]:
    """Entry point for uipath run agent command.

    Args:
        agent_name: Name of the agent to use
        agent_input: Either a dict with input data, or int index of datapoint to use (default: None)

    Returns:
        Single StateGraph for the specified input or datapoint to maintain CLI compatibility.
    """
    if debug:
        print(f"DEBUG: agent() called with agent_name={agent_name}, agent_input={agent_input}, type={type(agent_input)}")
    agent_scenario = get_agents()[agent_name]
    loop = BasicLoop(
        scenario=agent_scenario,
        llm=get_model(),
        print_trace=True,
        parallel_tool_calls=False,
        debug=debug,
    )
    if agent_input is None:
        agent_input = int(os.getenv("AGENT_INPUT", 0))
    if isinstance(agent_input, int):
        agent_input = agent_scenario.datapoints[agent_input].input
    graph = loop.build_graph(agent_input)
    yield graph


@asynccontextmanager
async def agents_with_datapoints(agent_name: str = "calculator") -> AsyncGenerator[List[tuple[StateGraph, Datapoint]], None]:
    """Create and return all LangGraph agents using the enhanced BasicLoop.

    Returns:
        A list of (StateGraph, Datapoint) tuples that can be executed.
    """
    agent_scenario = get_agents()[agent_name]
    loop = BasicLoop(
        scenario=agent_scenario,
        llm=get_model(),
        print_trace=True,
        parallel_tool_calls=False,
    )

    graphs = []
    for datapoint in agent_scenario.datapoints:
        graph = loop.build_graph(datapoint.input)
        graphs.append((graph, datapoint))

    yield graphs


@asynccontextmanager
async def calculator_agent() -> AsyncGenerator[StateGraph, None]:
    """Pre-configured calculator agent entry point that supports both chat and datapoint modes."""
    # Don't force any datapoint - let the runtime determine the mode
    async with agent(agent_name="calculator", agent_input=None) as graph:
        yield graph
