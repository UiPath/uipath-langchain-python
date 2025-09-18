"""Base evaluator abstract class for agent evaluation."""

from collections import Counter
from collections.abc import Callable, Coroutine
import functools
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from pydantic import BaseModel, ConfigDict

from gym_sample.evals_utils import AgentExecution, BooleanEvaluationResult, ErrorEvaluationResult, EvaluationResult, EvaluatorCategory, EvaluatorType, NumericEvaluationResult
from gym_sample.trace_utils import extract_tool_calls_names, extract_tool_calls, lcs_score, tool_args_score


def track_evaluation_metrics(func: Callable[..., Any]) -> Callable[[Any, Any], Coroutine[Any, Any, EvaluationResult]]:
    """Decorator to track evaluation metrics and handle errors gracefully."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> EvaluationResult:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            result = ErrorEvaluationResult(
                details="Exception thrown by evaluator: {}".format(e),
                evaluation_time=time.time() - start_time,
            )
        end_time = time.time()
        execution_time = end_time - start_time

        result.evaluation_time = execution_time
        return result

    return wrapper


T = TypeVar("T")


class BaseEvaluator(BaseModel, Generic[T], ABC):
    """Abstract base class for all evaluators."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    description: str
    target_output_key: str = "*"
    created_at: str
    updated_at: str
    category: EvaluatorCategory
    evaluator_type: EvaluatorType

    def __init_subclass__(cls, **kwargs: Any):
        """Hook for subclass creation - automatically applies evaluation metrics tracking."""
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "evaluate") and not getattr(
            cls.evaluate, "_has_metrics_decorator", False
        ):
            cls.evaluate = track_evaluation_metrics(cls.evaluate)  # type: ignore[method-assign]
            cls.evaluate._has_metrics_decorator = True  # type: ignore[attr-defined]

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for Pydantic models."""
        pass

    @abstractmethod
    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: T
    ) -> EvaluationResult:
        """Evaluate the given data and return a result.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - actual_output: The actual output from the agent
                - spans: The execution spans to use for the evaluation
            evaluation_criteria: The criteria to evaluate
        Returns:
            EvaluationResult containing the score and details
        """
        pass


class DeterministicEvaluatorBase(BaseEvaluator[T], ABC):
    """Base class for evaluators that produce deterministic, reproducible results.
    This class provides utility methods for canonical JSON comparison and number normalization
    to ensure consistent evaluation results across runs.
    """

    def _canonical_json(self, obj: Any) -> str:
        """Convert an object to canonical JSON string for consistent comparison.
        Args:
            obj: The object to convert to canonical JSON
        Returns:
            str: Canonical JSON string with normalized numbers and sorted keys
        """
        return json.dumps(
            self._normalize_numbers(obj),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _normalize_numbers(self, obj: Any) -> Any:
        """Recursively normalize numbers in nested data structures.
        Converts all numeric values (int, float) to float for consistent comparison,
        while preserving booleans and other data types.
        Args:
            obj: The object to normalize
        Returns:
            Any: Object with normalized numbers
        """
        if isinstance(obj, dict):
            return {k: self._normalize_numbers(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._normalize_numbers(v) for v in obj]
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return float(obj)
        return obj


class ExactMatchEvaluator(DeterministicEvaluatorBase[dict[str, Any]]):
    """Evaluator that performs exact structural matching between expected and actual outputs.
    This evaluator returns True if the actual output exactly matches the expected output
    after canonical JSON normalization, and False otherwise. Numbers are normalized
    to floats for consistent comparison.
    """

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate whether actual output exactly matches expected output.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - actual_output: The actual output from the agent
                - spans: The execution spans to use for the evaluation
            evaluation_criteria: The criteria to evaluate
        Returns:
            EvaluationResult: Boolean result indicating exact match (True/False)
        """
        return BooleanEvaluationResult(
            score=self._canonical_json(agent_execution.agent_output)
            == self._canonical_json(evaluation_criteria)
        )


class ToolCallOrderEvaluator(DeterministicEvaluatorBase[list[str]]):
    """Evaluator that checks if the tool calls are in the correct order.
    This evaluator returns True if the tool calls are in the correct order, and False otherwise.
    """

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: list[str]
    ) -> EvaluationResult:
        """Evaluate if the tool calls are in the correct order.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - actual_output: The actual output from the agent
                - spans: The execution spans to use for the evaluation
            evaluation_criteria: The criteria to evaluate
        Returns:
            EvaluationResult: Boolean result indicating correct tool call order (True/False)
        """
        tool_calls_order = extract_tool_calls_names(agent_execution.agent_trace)
        return NumericEvaluationResult(
            score=lcs_score(tool_calls_order, evaluation_criteria)
        )


class ToolCallCountEvaluator(DeterministicEvaluatorBase[Dict[str, int]]):
    """Evaluator that checks if the tool calls are in the correct order.
    This evaluator returns True if the tool calls are in the correct order, and False otherwise.
    """

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: Dict[str, int]
    ) -> EvaluationResult:
        """Evaluate if the tool calls are in the correct order.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - actual_output: The actual output from the agent
                - spans: The execution spans to use for the evaluation
            evaluation_criteria: The criteria to evaluate
        Returns:
            EvaluationResult: Boolean result indicating correct tool call order (True/False)
        """
        tool_calls_names = extract_tool_calls_names(agent_execution.agent_trace)
        return NumericEvaluationResult(
            score=float(Counter(tool_calls_names) == Counter(evaluation_criteria))
        )


class ToolCallArgumentsEvaluator(DeterministicEvaluatorBase[List[Dict[str, Any]]]):
    """Evaluator that checks the correctness of the arguments of the tool calls
    The order does not matter for this evaluator.

    Args:
        agent_execution: The execution details containing:
            - agent_input: The input received by the agent
            - actual_output: The actual output from the agent
            - spans: The execution spans to use for the evaluation
        evaluation_criteria: A dictionary of tool call names and their expected arguments.

    Returns:
        EvaluationResult: Boolean result indicating correct tool call arguments (True/False)
    """

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate if the tool calls are in the correct order.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - actual_output: The actual output from the agent
                - spans: The execution spans to use for the evaluation
            evaluation_criteria: The criteria to evaluate
        Returns:
            EvaluationResult: Boolean result indicating correct tool call order (True/False)
        """
        tool_calls = extract_tool_calls(agent_execution.agent_trace)
        return NumericEvaluationResult(
            score=tool_args_score(tool_calls, evaluation_criteria)
        )
