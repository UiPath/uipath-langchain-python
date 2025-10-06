from typing import List
from uipath.eval.evaluators import BaseEvaluator
from eval.coded_evaluators import (
    BaseEvaluator as CodedBaseEvaluator,
    ExactMatchEvaluator as CodedExactMatchEvaluator,
    ContainsEvaluator as CodedContainsEvaluator,
    JsonSimilarityEvaluator as CodedJsonSimilarityEvaluator,
    ToolCallOrderEvaluator as CodedToolCallOrderEvaluator,
    ToolCallCountEvaluator as CodedToolCallCountEvaluator,
    ToolCallArgsEvaluator as CodedToolCallArgsEvaluator,
    ToolCallOutputEvaluator as CodedToolCallOutputEvaluator,
    LLMJudgeOutputEvaluator as CodedLLMJudgeOutputEvaluator,
    LLMJudgeStrictJSONSimilarityOutputEvaluator as CodedLLMJudgeStrictJSONSimilarityOutputEvaluator,
    LLMJudgeTrajectoryEvaluator as CodedLLMJudgeTrajectoryEvaluator,
    LLMJudgeSimulationTrajectoryEvaluator as CodedLLMJudgeSimulationTrajectoryEvaluator,
)

def get_calculator_evaluators(include_llm_judge: bool = False) -> List[BaseEvaluator | CodedBaseEvaluator]:
    """Create evaluators using the new CodedEvaluator approach.
    """
    evaluators: List[BaseEvaluator | CodedBaseEvaluator] = [
        CodedExactMatchEvaluator.model_validate({"config": {"negated": False}}),
        CodedContainsEvaluator.model_validate({"config": {"negated": False}}),
        CodedJsonSimilarityEvaluator.model_validate({"config": {}}),
        CodedToolCallOrderEvaluator.model_validate({
            "config": {
                "strict": False,
            },
        }),
        CodedToolCallCountEvaluator.model_validate({
            "config": {
                "strict": False,
            },
        }),
        CodedToolCallArgsEvaluator.model_validate({
            "config": {
                "strict": False,
                "subset": False,
            },
        }),
        CodedToolCallOutputEvaluator.model_validate({
            "config": {
                "strict": False,
            },
        }),
    ]

    if include_llm_judge:
        evaluators.extend([
            CodedLLMJudgeOutputEvaluator.model_validate({
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
            CodedLLMJudgeStrictJSONSimilarityOutputEvaluator.model_validate({
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
            CodedLLMJudgeTrajectoryEvaluator.model_validate({
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
            CodedLLMJudgeSimulationTrajectoryEvaluator.model_validate({
                "config": {
                    "model": "gpt-4o-2024-11-20",
                    "temperature": 0.0,
                },
            }),
        ])

    return evaluators
