from typing import List
from uipath.eval.evaluators import LegacyBaseEvaluator
from uipath.eval.evaluators import (
    BaseEvaluator,
    ExactMatchEvaluator,
    ContainsEvaluator,
    ToolCallOrderEvaluator,
    ToolCallCountEvaluator,
    ToolCallArgsEvaluator,
    LLMJudgeOutputEvaluator,
)

def get_evaluators(include_llm_judge: bool = False) -> List[LegacyBaseEvaluator | BaseEvaluator]:
    """Create evaluators for loan agent.

    Evaluators match the test case metrics:
    - ExactMatchEvaluator: Validates ActionCenterTaskCreated boolean
    - ContainsEvaluator: Checks ExecutionDetails contains expected keywords (e.g., "loan", "Applied")
    - ToolCallOrderEvaluator: Validates tool calls are in the expected sequence
    - ToolCallCountEvaluator: Validates the count of each tool call
    - ToolCallArgsEvaluator: Validates tool call arguments match expected values
    - LLMJudgeOutputEvaluator: Uses LLM to judge ExecutionDetails factuality (optional)
    """
    evaluators: List[LegacyBaseEvaluator | BaseEvaluator] = [
        # Validates ActionCenterTaskCreated matches expected boolean value
        ExactMatchEvaluator.model_validate({
            "id": "ExactMatchEvaluator",
            "config": {
                "negated": False,
                "target_output_key": "ActionCenterTaskCreated"
            }
        }),
        # Validates ExecutionDetails contains common keywords found across scenarios
        ContainsEvaluator.model_validate({
            "id": "ContainsEvaluator",
            "config": {
                "negated": False,
                "target_output_key": "ExecutionDetails"
            }
        }),
        # Validates tool calls occur in the expected order
        ToolCallOrderEvaluator.model_validate({
            "id": "ToolCallOrderEvaluator",
            "config": {
                "strict": False,
            },
        }),
        # Validates the count of each tool call matches expectations
        ToolCallCountEvaluator.model_validate({
            "id": "ToolCallCountEvaluator",
            "config": {
                "strict": False,
            },
        }),
        # Validates tool call arguments match expected values
        ToolCallArgsEvaluator.model_validate({
            "id": "ToolCallArgsEvaluator",
            "config": {
                "strict": False,
                "subset": False,
            },
        }),
    ]

    if include_llm_judge:
        # LLM Judge evaluator for ExecutionDetails - matches LLMJudgeFactualityLoanAgent metric
        evaluators.extend([
            LLMJudgeOutputEvaluator.model_validate({
                "id": "LLMJudgeOutputEvaluator",
                "config": {
                    "target_output_key": "ExecutionDetails",
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
        ])

    return evaluators
