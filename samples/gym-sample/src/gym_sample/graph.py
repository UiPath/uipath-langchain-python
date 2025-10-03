import os
from typing import Any, AsyncGenerator, Dict, List
from contextlib import asynccontextmanager

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from pydantic import BaseModel
from gym_sample.uipath_gym_types import (
    AgentBaseClass,
    EndExecutionTool,
    StructuredTool,
    BasicLoop,
    Datapoint,
)
from uipath_langchain.chat import UiPathAzureChatOpenAI
from gym_sample.calculator.agent import get_calculator_agent


def get_model() -> BaseChatModel:
    """Get the ChatAnthropic model (created lazily to allow environment loading)."""
    return UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")


def get_agents() -> Dict[str, AgentBaseClass]:
    """Get the agents (created lazily to allow environment loading)."""
    return {
        "calculator": get_calculator_agent(),
    }


@asynccontextmanager
async def agents_with_datapoints(agent_name: str = "calculator") -> AsyncGenerator[List[tuple[StateGraph, Datapoint]], None]:
    """Create and return all LangGraph agents for evaluation mode.

    Each graph pre-binds its datapoint input at build time.

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
        # Build evaluation graph with pre-bound input
        graph = loop.build_evaluation_graph(datapoint.input)
        graphs.append((graph, datapoint))

    yield graphs


@asynccontextmanager
async def agent(agent_name: str = "calculator") -> AsyncGenerator[StateGraph, None]:
    """Create and return a LangGraph agent for evaluation mode.

    Returns:
        A StateGraph that can be executed.
    """
    agent_scenario = get_agents()[agent_name]
    loop = BasicLoop(
        scenario=agent_scenario,
        llm=get_model(),
        print_trace=True,
        parallel_tool_calls=False,
        debug=False,
    )

    # Build CLI graph that accepts input at runtime
    graph = loop.build_cli_graph()
    yield graph


@asynccontextmanager
async def calculator_agent() -> AsyncGenerator[StateGraph, None]:
    """Pre-configured calculator agent entry point for CLI usage.

    Following the ticket-classification pattern:
    - Graph uses input=CalculatorInput, output=CalculatorOutput
    - Accepts typed input at runtime via graph.ainvoke(CalculatorInput(...))
    - CLI calls with agent_input=None and passes input at graph invocation

    Example: uipath run calculator '{"expression": "2 + 2"}'

    Args:
        agent_name: The name of the agent to create.

    Returns:
        StateGraph configured for CLI mode (accepts runtime input).
    """
    async with agent("calculator") as graph:
        yield graph
