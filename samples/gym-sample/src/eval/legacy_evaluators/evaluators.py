"""Base evaluator abstract class for agent evaluation."""

from collections import Counter
from collections.abc import Callable
from datetime import datetime
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator
from uipath._utils.constants import COMMUNITY_agents_SUFFIX
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
)
from uipath.eval.models import NumericEvaluationResult
from uipath.eval.models.models import EvaluatorCategory, EvaluatorType
from eval.legacy_evaluators.evaluators_helpers import AgentExecution, extract_tool_calls_names, extract_tool_calls, extract_tool_calls_outputs, tool_calls_count_score, tool_calls_order_score, tool_args_score, tool_output_score, trace_to_str
from uipath.eval.evaluators.deterministic_evaluator_base import (
    DeterministicEvaluatorBase,
)
from eval.legacy_evaluators.llm_judge_types import LLMJudgeOutputSchema, LLMJudgeStrictJSONSimilarityOutputSchema, LLMJudgeTrajectoryOutputSchema, PromptTemplates
from uipath.eval.evaluators.llm_as_judge_evaluator import LLMResponse
from eval.coded_evaluators import BaseEvaluator as CodedBaseEvaluator


class ExactMatchEvaluator(DeterministicEvaluatorBase[dict[str, Any]]):
    """Evaluator that performs exact structural matching between expected and actual outputs.

    This evaluator returns True if the actual output exactly matches the expected output
    after canonical JSON normalization, and False otherwise. Numbers are normalized
    to floats for consistent comparison.
    """

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate whether actual output exactly matches expected output.

        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - actual_output: The actual output from the agent
                - spans: The execution spans to use for the evaluation
            evaluation_criteria: The criteria dict with 'expected_output' key

        Returns:
            EvaluationResult: Boolean result indicating exact match (True/False)
        """
        expected_output = evaluation_criteria.get("expected_output", {})
        return NumericEvaluationResult(
            score=float(self._canonical_json(agent_execution.agent_output) == self._canonical_json(expected_output))
        )



class ToolCallOrderEvaluator(DeterministicEvaluatorBase[Dict[str, Any]]):
    """Evaluator that checks if the tool calls are in the correct order.
    This evaluator returns True if the tool calls are in the correct order, and False otherwise.
    """
    strict: bool = False

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate if the tool calls are in the correct order.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
            evaluation_criteria: The criteria dict with 'tool_calls_order' key
        Returns:
            EvaluationResult: Boolean result indicating correct tool call order (True/False)
        """
        # Extract the tool_calls_order from the structured criteria
        expected_order = evaluation_criteria.get("tool_calls_order", [])
        tool_calls_order = extract_tool_calls_names(agent_execution.agent_trace)
        return NumericEvaluationResult(
            score=tool_calls_order_score(tool_calls_order, expected_order, self.strict)
        )


class ToolCallCountEvaluator(DeterministicEvaluatorBase[Dict[str, Any]]):
    """Evaluator that checks if the tool calls are in the correct order.
    This evaluator returns True if the tool calls are in the correct order, and False otherwise.
    """
    strict: bool = False

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate if the tool calls are in the correct order.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
            evaluation_criteria: The criteria dict with 'tool_calls_count' key
        Returns:
            EvaluationResult: Boolean result indicating correct tool call order (True/False)
        """
        # Extract the tool_calls_count from the structured criteria
        expected_counts = evaluation_criteria.get("tool_calls_count", {})
        tool_calls_count = Counter(extract_tool_calls_names(agent_execution.agent_trace))
        return NumericEvaluationResult(
            score=tool_calls_count_score(tool_calls_count, expected_counts, self.strict)
        )


class ToolCallArgumentsEvaluator(DeterministicEvaluatorBase[Dict[str, Any]]):
    """Evaluator that checks the correctness of the arguments of the tool calls
    The order does not matter for this evaluator.

    Args:
        agent_execution: The execution details containing:
            - agent_input: The input received by the agent
            - agent_output: The final output of the agent
            - agent_trace: The execution spans to use for the evaluation
        evaluation_criteria: A dictionary with 'tool_calls' key containing expected tool calls.

    Returns:
        EvaluationResult: Boolean result indicating correct tool call arguments (True/False)
    """
    strict: bool = False
    subset: bool = False

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate if the tool calls are in the correct order.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
            evaluation_criteria: The criteria dict with 'tool_calls' key
        Returns:
            EvaluationResult: Boolean result indicating correct tool call order (True/False)
        """
        # Extract the tool_calls from the structured criteria
        expected_tool_calls = evaluation_criteria.get("tool_calls", [])
        tool_calls = extract_tool_calls(agent_execution.agent_trace)
        return NumericEvaluationResult(
            score=tool_args_score(tool_calls, expected_tool_calls, self.strict, self.subset)
        )


class ToolCallOutputEvaluator(DeterministicEvaluatorBase[Dict[str, Any]]):
    """Evaluator that checks the correctness of the output of the tool calls
    The order does not matter for this evaluator.
    """
    strict: bool = False

    async def evaluate(
        self, agent_execution: AgentExecution, evaluation_criteria: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate if the tool calls are in the correct order.
        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
            evaluation_criteria: The criteria dict with 'tool_outputs' key
        Returns:
            EvaluationResult: Boolean result indicating correct tool call output (True/False)
        """
        # Extract the tool_outputs from the structured criteria
        expected_tool_outputs = evaluation_criteria.get("tool_outputs", [])
        tool_calls = extract_tool_calls_outputs(agent_execution.agent_trace)
        return NumericEvaluationResult(
            score=tool_output_score(tool_calls, expected_tool_outputs, self.strict)
        )


class LLMJudgeEvaluator(BaseEvaluator[str | Dict[str, Any]]):
    """Evaluator that uses an LLM to judge the quality of agent output."""

    model: str
    prompt: str = PromptTemplates.LLM_JUDGE_DEFAULT_USER_PROMPT
    system_prompt: str = PromptTemplates.LLM_JUDGE_SYSTEM_PROMPT
    output_schema: type[BaseModel] = LLMJudgeOutputSchema
    actual_output_placeholder: str = "{{ActualOutput}}"
    evaluation_criteria_placeholder: str = "{{ExpectedOutput}}"
    llm_service: Optional[Callable] = None
    temperature: float = 0.0
    max_tokens: int = 1000

    @model_validator(mode="after")
    def validate_prompt_placeholders(self) -> "LLMJudgeEvaluator":
        """Validate that prompt contains required placeholders."""
        if self.actual_output_placeholder not in self.prompt or self.evaluation_criteria_placeholder not in self.prompt:
            raise ValueError(
                f"Prompt must contain both {self.actual_output_placeholder} and {self.evaluation_criteria_placeholder} placeholders"
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize the LLM service if not provided."""
        super().model_post_init(__context)
        if self.llm_service is None:
            self.llm_service = self._get_llm_service()

    def _get_llm_service(self):
        """Get the LLM service from the UiPath instance."""
        from uipath import UiPath

        uipath = UiPath()
        return uipath.llm.chat_completions

    async def evaluate(
        self,
        agent_execution: AgentExecution,
        evaluation_criteria: str | Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate using an LLM as a judge.

        Sends the formatted prompt to the configured LLM and expects a JSON response
        with a numerical score (0-100) and justification.

            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
            evaluation_criteria: The criteria to evaluate

        Returns:
            EvaluationResult: Numerical score with LLM justification as details
        """
        # Create the evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(
            agent_execution=agent_execution,
            evaluation_criteria=evaluation_criteria,
        )

        llm_response = await self._get_llm_response(evaluation_prompt)

        return NumericEvaluationResult(
            score=round(llm_response.score / 100.0, 2),
            details=llm_response.justification,
        )

    def _get_actual_output(self, agent_execution: AgentExecution) -> str | Dict[str, Any]:
        """Get the actual output from the agent execution."""
        return agent_execution.agent_output

    def _create_evaluation_prompt(
        self, agent_execution: AgentExecution, evaluation_criteria: str | Dict[str, Any]
    ) -> str:
        """Create the evaluation prompt for the LLM."""
        formatted_prompt = self.prompt.replace(
            self.actual_output_placeholder,
            str(self._get_actual_output(agent_execution)),
        )
        expected_output = evaluation_criteria.get("expected_output", {}) if isinstance(evaluation_criteria, dict) else evaluation_criteria
        formatted_prompt = formatted_prompt.replace(
            self.evaluation_criteria_placeholder,
            str(expected_output),
        )

        return formatted_prompt

    async def _get_llm_response(self, evaluation_prompt: str) -> LLMResponse:
        """Get response from the LLM.

        Args:
            evaluation_prompt: The formatted prompt to send to the LLM

        Returns:
            LLMResponse with score and justification
        """
        # remove community-agents suffix from llm model name
        model = self.model
        if model.endswith(COMMUNITY_agents_SUFFIX):
            model = model.replace(COMMUNITY_agents_SUFFIX, "")

        # Prepare the request
        request_data = {
            "model": model,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": evaluation_prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation_response",
                    "schema": self.output_schema.model_json_schema(),
                },
            },
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        assert self.llm_service is not None, "LLM service not initialized"
        response = await self.llm_service(**request_data)
        return LLMResponse(**json.loads(str(response.choices[-1].message.content)))


class LLMJudgeStrictJSONSimilarityEvaluator(LLMJudgeEvaluator):
    """Evaluator that uses an LLM to judge the quality of agent output."""

    prompt: str = PromptTemplates.LLM_JUDGE_STRICT_JSON_SIMILARITY_DEFAULT_USER_PROMPT
    system_prompt: str = PromptTemplates.LLM_JUDGE_STRICT_JSON_SIMILARITY_SYSTEM_PROMPT
    output_schema: type[BaseModel] = LLMJudgeStrictJSONSimilarityOutputSchema


class LLMJudgeTrajectoryEvaluator(LLMJudgeEvaluator):
    """Evaluator that uses an LLM to judge the quality of agent output."""

    prompt: str = PromptTemplates.LLM_JUDGE_TRAJECTORY_DEFAULT_USER_PROMPT
    system_prompt: str = PromptTemplates.LLM_JUDGE_TRAJECTORY_SYSTEM_PROMPT
    output_schema: type[BaseModel] = LLMJudgeTrajectoryOutputSchema
    actual_output_placeholder: str = "{{AgentRunHistory}}"
    evaluation_criteria_placeholder: str = "{{ExpectedAgentBehavior}}"
    user_input_placeholder: str = "{{UserOrSyntheticInput}}"
    simulation_instructions_placeholder: str = "{{SimulationInstructions}}"

    def _get_actual_output(self, agent_execution: AgentExecution) -> str | Dict[str, Any]:
        """Get the actual output from the agent execution."""
        return trace_to_str(agent_execution.agent_trace)

    def _create_evaluation_prompt(
        self, agent_execution: AgentExecution, evaluation_criteria: str | Dict[str, Any]
    ) -> str:
        """Create the evaluation prompt for the LLM."""
        formatted_prompt = super()._create_evaluation_prompt(agent_execution, evaluation_criteria)
        formatted_prompt = formatted_prompt.replace(
            self.user_input_placeholder,
            str(agent_execution.agent_input),
        )
        formatted_prompt = formatted_prompt.replace(
            self.simulation_instructions_placeholder,
            agent_execution.simulation_instructions,
        )
        return formatted_prompt


class LLMJudgeSimulationTrajectoryEvaluator(LLMJudgeTrajectoryEvaluator):
    """Evaluator that uses an LLM to judge the quality of agent output."""

    prompt: str = PromptTemplates.LLM_JUDGE_SIMULATION_TRAJECTORY_DEFAULT_USER_PROMPT
    system_prompt: str = PromptTemplates.LLM_JUDGE_SIMULATION_TRAJECTORY_SYSTEM_PROMPT


def create_old_evaluators(include_llm_judge: bool = False) -> List[BaseEvaluator | CodedBaseEvaluator]:
    """Create evaluators using the old BaseEvaluator approach.

    Returns:
        List of evaluators.
    """
    now = datetime.now().isoformat()

    exact_match_evaluator = ExactMatchEvaluator(
        id="ExactMatchEvaluator",
        name="ExactMatchEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates if the actual output exactly matches the expected output",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Equals,
    )

    tool_call_order_evaluator = ToolCallOrderEvaluator(
        id="ToolCallOrderEvaluator",
        name="ToolCallOrderEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct order",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
    )

    tool_call_count_evaluator = ToolCallCountEvaluator(
        id="ToolCallCountEvaluator",
        name="ToolCallCountEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct count",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
    )

    tool_call_arguments_evaluator = ToolCallArgumentsEvaluator(
        id="ToolCallArgsEvaluator",
        name="ToolCallArgsEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct arguments",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
        subset=False,
    )

    tool_call_output_evaluator = ToolCallOutputEvaluator(
        id="ToolCallOutputEvaluator",
        name="ToolCallOutputEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct output",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
    )

    llm_judge_evaluator = LLMJudgeEvaluator(
        id="LLMJudgeOutputEvaluator",
        name="LLMJudgeOutputEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.LlmAsAJudge,
        evaluator_type=EvaluatorType.Custom,
        model="gpt-4o-2024-11-20",
    )

    llm_judge_strict_json_similarity_evaluator = LLMJudgeStrictJSONSimilarityEvaluator(
        id="LLMJudgeStrictJSONSimilarityOutputEvaluator",
        name="LLMJudgeStrictJSONSimilarityOutputEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.LlmAsAJudge,
        evaluator_type=EvaluatorType.Custom,
        model="gpt-4o-2024-11-20",
    )

    llm_judge_trajectory_evaluator = LLMJudgeTrajectoryEvaluator(
        id="LLMJudgeTrajectoryEvaluator",
        name="LLMJudgeTrajectoryEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.LlmAsAJudge,
        evaluator_type=EvaluatorType.Custom,
        model="gpt-4o-2024-11-20",
    )

    llm_judge_simulation_trajectory_evaluator = LLMJudgeSimulationTrajectoryEvaluator(
        id="LLMJudgeSimulationEvaluator",
        name="LLMJudgeSimulationEvaluator",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.Trajectory,
        evaluator_type=EvaluatorType.Trajectory,
        model="gpt-4o-2024-11-20",
    )

    evaluators = [exact_match_evaluator, tool_call_order_evaluator, tool_call_count_evaluator, tool_call_arguments_evaluator, tool_call_output_evaluator]
    if include_llm_judge:
        evaluators.extend([llm_judge_evaluator, llm_judge_strict_json_similarity_evaluator, llm_judge_trajectory_evaluator, llm_judge_simulation_trajectory_evaluator])

    return evaluators
