import json
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Generator

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from uipath._cli._evals._evaluator_factory import EvaluatorFactory
from uipath._cli._evals._models._evaluation_set import EvaluationItem
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.llm_as_judge_evaluator import LLMJudgeMixin
from uipath.eval.models import LLMResponse
from uipath_langchain.chat.models import UiPathAzureChatOpenAI

from gym_sample.calculator.agent import get_calculator_agent
from gym_sample.loan.agent import get_loan_agent
from gym_sample.loan.evals import get_evaluators as get_loan_evaluators
from gym_sample.uipath_gym_types import (
    AgentBaseClass,
    BasicLoop,
)


def get_model(model_name: str) -> BaseChatModel:
    return UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20")


def get_agents() -> Dict[str, AgentBaseClass]:
    """Get the agents (created lazily to allow environment loading)."""
    return {
        "calculator": get_calculator_agent(),
        "loan": get_loan_agent(),
    }


def get_llm_response_factory(model: BaseChatModel) -> Any:
    async def get_llm_response(evaluation_prompt: str) -> LLMResponse:
        response = await model.with_structured_output(LLMResponse).ainvoke(
            evaluation_prompt
        )
        return LLMResponse.model_validate(response)

    return get_llm_response


def get_evaluators(
    agent_dir: Path, override_judge_model: bool = False
) -> dict[str, BaseEvaluator[Any, Any, Any]]:
    evaluators: dict[str, BaseEvaluator[Any, Any, Any]] = {}
    for file in (agent_dir / "evaluators").glob("evaluator-*.json"):
        with open(file, "r") as f:
            spec = json.load(f)
        evaluator = EvaluatorFactory.create_evaluator(
            spec,
            evaluators_dir=(agent_dir / "evaluators" / "custom"),
        )
        # Monkey patch the evaluator to use Gym Throttled models
        if isinstance(evaluator, LLMJudgeMixin):
            judge_model = (
                get_model(evaluator.config["model"])
                if not override_judge_model
                else get_model("gpt-4o-2024-11-20")
            )
            evaluator._get_llm_response = get_llm_response_factory(judge_model)
        evaluators[evaluator.id] = evaluator
    return evaluators


def get_all_evaluators() -> Dict[str, Callable[[bool], dict[str, BaseEvaluator[Any, Any, Any]]]]:
    """Get the evaluators (created lazily to allow environment loading)."""
    return {
        "calculator": lambda _: get_evaluators(Path(__file__).parent / "calculator"),
        "loan": lambda include_llm_judge: get_loan_evaluators(include_llm_judge),
    }


def agents_with_datapoints_generator(
    agent_name: str = "calculator",
    max_datapoints: int = 0,
) -> tuple[Generator[tuple[StateGraph[Any, Any], EvaluationItem], None, None], int]:
    """Create and return all LangGraph agents for evaluation mode.

    Uses the unified graph that accepts input at runtime.
    Each graph gets a fresh BasicLoop instance to avoid LLM state accumulation.

    Returns:
        A list of (StateGraph, Datapoint) tuples that can be executed.
    """
    agent_scenario = get_agents()[agent_name]
    num_datapoints = (
        len(agent_scenario.evaluation_set.evaluations)
        if max_datapoints == 0
        else max_datapoints
    )

    def generator() -> Generator[tuple[StateGraph[Any, Any], EvaluationItem], None, None]:
        # Create a fresh BasicLoop for each datapoint to avoid LLM state accumulation
        loop = BasicLoop(
            scenario=agent_scenario,
            llm=get_model("gpt-4o-2024-11-20"),
            print_trace=True,
            parallel_tool_calls=False,
        )
        # Build unified graph that accepts input at runtime
        graph = loop.build_graph()
        datapoints = agent_scenario.evaluation_set.evaluations
        if max_datapoints > 0:
            datapoints = islice(datapoints, max_datapoints)
        for datapoint in datapoints:
            yield (graph, datapoint)

    return generator(), num_datapoints
