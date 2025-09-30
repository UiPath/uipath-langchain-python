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
from gym_sample.evals import get_datapoints


def get_model() -> BaseChatModel:
    """Get the ChatAnthropic model (created lazily to allow environment loading)."""
    return UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")

class CalculatorInput(BaseModel):
    expression: str

class CalculatorOutput(BaseModel):
    answer: float


def get_agents() -> Dict[str, AgentBaseClass]:
    """Get the agents (created lazily to allow environment loading)."""
    return {
        "calculator":
            AgentBaseClass(
                system_prompt="You are a calculator agent. You can perform mathematical operations using the available tools. When you have completed the calculation, use the end_execution tool to provide your final result with a score (0.0 to 1.0 representing confidence) and observations about the calculation process.",
                user_prompt="Calculate the result of: {expression}.",
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
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
async def calculator_agent(
    agent_input: CalculatorInput | None = None
) -> AsyncGenerator[StateGraph, None]:
    """Pre-configured calculator agent entry point for CLI usage.

    Following the ticket-classification pattern:
    - Graph uses input=CalculatorInput, output=CalculatorOutput
    - Accepts typed input at runtime via graph.ainvoke(CalculatorInput(...))
    - CLI calls with agent_input=None and passes input at graph invocation

    Example: uipath run calculator '{"expression": "2 + 2"}'

    Args:
        agent_input: Not used - kept for API compatibility

    Returns:
        StateGraph configured for CLI mode (accepts runtime input).
    """
    agent_scenario = get_agents()["calculator"]
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
