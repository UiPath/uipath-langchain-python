"""Base evaluator abstract class for agent evaluation."""

from collections import Counter
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator
from uipath._services import UiPathLlmChatService
from uipath._utils.constants import COMMUNITY_agents_SUFFIX
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
)
from uipath.eval.models import NumericEvaluationResult
from gym_sample.evals_helpers import AgentExecution, extract_tool_calls_names, extract_tool_calls, tool_calls_count_score, tool_calls_order_score, tool_args_score, trace_to_str
from uipath.eval.evaluators.deterministic_evaluator_base import (
    DeterministicEvaluatorBase,
)
from gym_sample.llm_judge_types import LLMJudgeOutputSchema, LLMJudgeStrictJSONSimilarityOutputSchema, LLMJudgeTrajectoryOutputSchema, PromptTemplates
from uipath.eval.evaluators.llm_as_judge_evaluator import LLMResponse

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
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
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
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
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
            - agent_output: The final output of the agent
            - agent_trace: The execution spans to use for the evaluation
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
                - agent_output: The final output of the agent
                - agent_trace: The execution spans to use for the evaluation
            evaluation_criteria: The criteria to evaluate
        Returns:
            EvaluationResult: Boolean result indicating correct tool call order (True/False)
        """
        tool_calls = extract_tool_calls(agent_execution.agent_trace)
        return NumericEvaluationResult(
            score=tool_args_score(tool_calls, evaluation_criteria, self.strict, self.subset)
        )


class LLMJudgeEvaluator(BaseEvaluator[str | Dict[str, Any]]):
    """Evaluator that uses an LLM to judge the quality of agent output."""

    model: str
    prompt: str = PromptTemplates.LLM_JUDGE_DEFAULT_USER_PROMPT
    system_prompt: str = PromptTemplates.LLM_JUDGE_SYSTEM_PROMPT
    output_schema: type[BaseModel] = LLMJudgeOutputSchema
    actual_output_placeholder: str = "{{ActualOutput}}"
    evaluation_criteria_placeholder: str = "{{ExpectedOutput}}"
    llm: Optional[UiPathLlmChatService] = None

    @model_validator(mode="after")
    def validate_prompt_placeholders(self) -> "LLMJudgeEvaluator":
        """Validate that prompt contains required placeholders."""
        if self.actual_output_placeholder not in self.prompt or self.evaluation_criteria_placeholder not in self.prompt:
            raise ValueError(
                f"Prompt must contain both {self.actual_output_placeholder} and {self.evaluation_criteria_placeholder} placeholders"
            )
        return self

    def model_post_init(self, __context: Any):
        """Initialize the LLM service after model creation."""
        super().model_post_init(__context)
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM used for evaluation."""
        from uipath import UiPath

        uipath = UiPath()
        self.llm = uipath.llm

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
            score=llm_response.score,
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
        formatted_prompt = formatted_prompt.replace(
            self.evaluation_criteria_placeholder,
            str(evaluation_criteria),
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
        }

        response = await self.llm.chat_completions(**request_data)  # type: ignore
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
