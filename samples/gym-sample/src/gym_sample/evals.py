"""Base evaluator abstract class for agent evaluation."""

from collections import Counter
from typing import Any, Dict, List

from uipath.eval.evaluators.base_evaluator import (
    AgentExecution,
    EvaluationResult,
)
from uipath.eval.models import NumericEvaluationResult
from gym_sample.evals_helpers import extract_tool_calls_names, extract_tool_calls, tool_calls_count_score, tool_calls_order_score, tool_args_score
from uipath.eval.evaluators.deterministic_evaluator_base import (
    DeterministicEvaluatorBase,
)

class ToolCallOrderEvaluator(DeterministicEvaluatorBase[List[str]]):
    """Evaluator that checks if the tool calls are in the correct order.
    This evaluator returns True if the tool calls are in the correct order, and False otherwise.
    """
    strict: bool = False

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: List[str]
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
            score=tool_calls_order_score(tool_calls_order, evaluation_criteria, self.strict)
        )


class ToolCallCountEvaluator(DeterministicEvaluatorBase[Dict[str, int | str]]):
    """Evaluator that checks if the tool calls are in the correct order.
    This evaluator returns True if the tool calls are in the correct order, and False otherwise.
    """
    strict: bool = False

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: Dict[str, int | str]
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
        tool_calls_count = Counter(extract_tool_calls_names(agent_execution.agent_trace))
        return NumericEvaluationResult(
            score=tool_calls_count_score(tool_calls_count, evaluation_criteria, self.strict)
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
    strict: bool = False
    subset: bool = False

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
            score=tool_args_score(tool_calls, evaluation_criteria, self.strict, self.subset)
        )
