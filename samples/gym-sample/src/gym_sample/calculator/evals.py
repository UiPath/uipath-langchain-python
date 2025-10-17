from typing import List
from uipath.eval.evaluators import LegacyBaseEvaluator
from uipath.eval.coded_evaluators import (
    BaseEvaluator,
    ExactMatchEvaluator,
    ContainsEvaluator,
    JsonSimilarityEvaluator,
    ToolCallOrderEvaluator,
    ToolCallCountEvaluator,
    ToolCallArgsEvaluator,
    ToolCallOutputEvaluator,
    LLMJudgeOutputEvaluator,
    LLMJudgeStrictJSONSimilarityOutputEvaluator,
    LLMJudgeTrajectoryEvaluator,
    LLMJudgeSimulationTrajectoryEvaluator,
)

def get_evaluators(include_llm_judge: bool = False) -> List[LegacyBaseEvaluator | BaseEvaluator]:
    """Create evaluators for calculator agent.
    """
    evaluators: List[LegacyBaseEvaluator | BaseEvaluator] = [
        ExactMatchEvaluator.model_validate({"id": "ExactMatchEvaluator", "config": {"negated": False}}),
        ContainsEvaluator.model_validate({"id": "ContainsEvaluator", "config": {"negated": False, "target_output_key": "answer"}}),
        JsonSimilarityEvaluator.model_validate({"id": "JsonSimilarityEvaluator", "config": {}}),
        ToolCallOrderEvaluator.model_validate({
            "id": "ToolCallOrderEvaluator",
            "config": {
                "strict": False,
            },
        }),
        ToolCallCountEvaluator.model_validate({
            "id": "ToolCallCountEvaluator",
            "config": {
                "strict": False,
            },
        }),
        ToolCallArgsEvaluator.model_validate({
            "id": "ToolCallArgsEvaluator",
            "config": {
                "strict": False,
                "subset": False,
            },
        }),
        ToolCallOutputEvaluator.model_validate({
            "id": "ToolCallOutputEvaluator",
            "config": {
                "strict": False,
            },
        }),
    ]

    if include_llm_judge:
        evaluators.extend([
            LLMJudgeOutputEvaluator.model_validate({
                "id": "LLMJudgeOutputEvaluator",
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
            LLMJudgeStrictJSONSimilarityOutputEvaluator.model_validate({
                "id": "LLMJudgeStrictJSONSimilarityOutputEvaluator",
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
            LLMJudgeTrajectoryEvaluator.model_validate({
                "id": "LLMJudgeTrajectoryEvaluator",
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
            LLMJudgeSimulationTrajectoryEvaluator.model_validate({
                "id": "LLMJudgeSimulationTrajectoryEvaluator",
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
        ])

    return evaluators
