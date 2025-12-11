from pathlib import Path

from uipath._cli._evals._models._evaluation_set import EvaluationSet
from uipath._cli._utils._eval_set import EvalHelpers


def get_evaluation_set(path: Path) -> EvaluationSet:
    eval_set, _ = EvalHelpers.load_eval_set(str(path))
    return eval_set
