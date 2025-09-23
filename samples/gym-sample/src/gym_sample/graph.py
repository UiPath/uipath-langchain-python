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
async def make_graph(agent_name: str = "calculator") -> AsyncGenerator[List[tuple[StateGraph, Datapoint]], None]:
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


# For uipath run agent compatibility - returns single graph for first datapoint
def agent(agent_input: Dict[str, Any] = {}) -> StateGraph:
    """Entry point for uipath run agent command.

    Returns a single StateGraph for the first datapoint to maintain CLI compatibility.
    """
    agent_scenario = get_agents()["calculator"]
    loop = BasicLoop(
        scenario=agent_scenario,
        llm=get_model(),
        print_trace=True,
        parallel_tool_calls=False,
    )

    # Use first datapoint for CLI compatibility
    first_datapoint = agent_scenario.datapoints[0] if agent_scenario.datapoints else Datapoint(
        input={},
        evaluation_criteria={},
        simulation_instructions=""
    )
    return loop.build_graph(first_datapoint.input)
