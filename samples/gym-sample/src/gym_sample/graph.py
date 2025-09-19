from typing import Any, AsyncGenerator, Dict
from contextlib import asynccontextmanager

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel
from gym_sample.uipath_gym_types import (
    AgentBaseClass,
    EndExecutionTool,
    StructuredTool,
    BasicLoop,
)

def get_model() -> ChatAnthropic:
    """Get the ChatAnthropic model (created lazily to allow environment loading)."""
    return ChatAnthropic(model_name="claude-3-5-sonnet-latest", timeout=60, stop=None)


def get_agent_scenario() -> AgentBaseClass:
    """Get the agent scenario (created lazily to allow environment loading)."""
    return AgentBaseClass(
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
    )


@asynccontextmanager
async def make_graph(agent_input: Dict[str, Any] = {}) -> AsyncGenerator[StateGraph, None]:
    """Create and return the LangGraph agent using the enhanced BasicLoop.

    Returns:
        A StateGraph that can be executed by the UiPath runtime.
    """
    loop = BasicLoop(
        scenario=get_agent_scenario(),
        llm=get_model(),
        print_trace=True,
        parallel_tool_calls=False,
    )

    # Build and return the graph
    graph = loop.build_graph(agent_input)
    yield graph
