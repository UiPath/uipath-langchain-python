from collections.abc import Callable
from typing import AsyncGenerator, Dict, List
from contextlib import asynccontextmanager

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from uipath.eval.evaluators import LegacyBaseEvaluator
from uipath.eval.evaluators import BaseEvaluator
from gym_sample.uipath_gym_types import (
    AgentBaseClass,
    BasicLoop,
    Datapoint,
)
from uipath_langchain.chat import UiPathAzureChatOpenAI
from gym_sample.calculator.agent import get_calculator_agent
from gym_sample.loan.agent import get_loan_agent
from gym_sample.calculator.evals import get_evaluators as get_calculator_evaluators
from gym_sample.loan.evals import get_evaluators as get_loan_evaluators


def get_model() -> BaseChatModel:
    """Get the LLM (created lazily to allow environment loading)."""
    return UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")


def get_agents() -> Dict[str, AgentBaseClass]:
    """Get the agents (created lazily to allow environment loading)."""
    return {
        "calculator": get_calculator_agent(),
        "loan": get_loan_agent(),
    }


def get_all_evaluators() -> Dict[str, Callable[[bool], List[LegacyBaseEvaluator | BaseEvaluator]]]:
    """Get the evaluators (created lazily to allow environment loading)."""
    return {
        "calculator": lambda include_llm_judge: get_calculator_evaluators(include_llm_judge),
        "loan": lambda include_llm_judge: get_loan_evaluators(include_llm_judge),
    }


@asynccontextmanager
async def agents_with_datapoints(agent_name: str = "calculator") -> AsyncGenerator[List[tuple[StateGraph, Datapoint]], None]:
    """Create and return all LangGraph agents for evaluation mode.

    Each graph pre-binds its datapoint input at build time.
    Each graph gets a fresh BasicLoop instance to avoid LLM state accumulation.

    Returns:
        A list of (StateGraph, Datapoint) tuples that can be executed.
    """
    agent_scenario = get_agents()[agent_name]

    graphs = []
    for datapoint in agent_scenario.datapoints:
        # Create a fresh BasicLoop for each datapoint to avoid LLM state accumulation
        loop = BasicLoop(
            scenario=agent_scenario,
            llm=get_model(),
            print_trace=True,
            parallel_tool_calls=False,
        )
        # Build evaluation graph with pre-bound input
        graph = loop.build_evaluation_graph(datapoint.input)
        graphs.append((graph, datapoint))

    yield graphs


async def agent(agent_name: str = "calculator") -> StateGraph:
    """Create and return a LangGraph agent for CLI mode.

    Args:
        agent_name: The name of the agent to create.

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
    return loop.build_cli_graph()


async def calculator_agent() -> StateGraph:
    """Pre-configured calculator agent entry point for CLI usage.

    Following the ticket-classification pattern:
    - Graph uses input=CalculatorInput, output=CalculatorOutput
    - Accepts typed input at runtime via graph.ainvoke(CalculatorInput(...))
    - CLI calls with agent_input=None and passes input at graph invocation

    Example: uipath run calculator '{"expression": "2 + 2"}'

    Returns:
        StateGraph configured for CLI mode (accepts runtime input).
    """
    return await agent("calculator")


async def loan_agent() -> StateGraph:
    """Pre-configured loan agent entry point for CLI usage.

    Returns:
        StateGraph configured for CLI mode (accepts runtime input).
    """
    return await agent("loan")
