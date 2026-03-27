"""Deterministic (rule-based) guardrail validator."""

import inspect
from typing import Any, Callable, ClassVar, Sequence

from uipath.core.guardrails import GuardrailScope, GuardrailValidationResult

from ...enums import GuardrailExecutionStage
from ._base import GuardrailValidatorBase

RuleFunction = (
    Callable[[dict[str, Any]], bool] | Callable[[dict[str, Any], dict[str, Any]], bool]
)
"""Type alias for deterministic rule functions.

A rule is a callable that returns ``True`` to signal a violation, ``False``
to pass. It accepts either:

- One parameter: ``(input_dict) -> bool`` — evaluated at PRE stage.
- Two parameters: ``(input_dict, output_dict) -> bool`` — evaluated at POST stage.

All rules must detect a violation for the guardrail action to trigger (AND
semantics). An empty rules list always triggers the action.
"""


class DeterministicValidator(GuardrailValidatorBase):
    """Validates tool input/output using local Python rule functions.

    No UiPath API call is made. Rules run in-process, making this suitable
    for fast, deterministic checks such as keyword filtering, length limits,
    or regex matching.

    Restricted to TOOL scope only.

    Args:
        rules: Sequence of rule callables. Each rule receives the tool input
            dict (1-parameter rules) or both input and output dicts
            (2-parameter rules). Returns ``True`` to flag a violation.
            ALL rules must flag a violation for the action to trigger.
            An empty list always triggers the action.

    Raises:
        ValueError: If any rule is not callable or has an unsupported parameter count.

    Example::

        donkey_filter = DeterministicValidator(
            rules=[lambda args: "donkey" in args.get("joke", "").lower()]
        )

        @guardrail(
            validator=donkey_filter,
            action=CustomFilterAction(word_to_filter="donkey", replacement="[censored]"),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Word Filter",
        )
        @tool
        def analyze_joke_syntax(joke: str) -> str: ...
    """

    supported_scopes = [GuardrailScope.TOOL]
    supported_stages: ClassVar[list[GuardrailExecutionStage]] = []  # all stages allowed

    def __init__(self, rules: Sequence[RuleFunction] = ()) -> None:
        for i, rule in enumerate(rules):
            if not callable(rule):
                raise ValueError(f"Rule {i + 1} must be callable, got {type(rule)}")
            sig = inspect.signature(rule)
            param_count = len(sig.parameters)
            if param_count not in (1, 2):
                raise ValueError(
                    f"Rule {i + 1} must have 1 or 2 parameters, got {param_count}"
                )
        self.rules = list(rules)

    def evaluate(
        self,
        data: str | dict[str, Any],
        stage: GuardrailExecutionStage,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
    ) -> GuardrailValidationResult:
        """Evaluate rules locally against tool input/output dicts.

        Args:
            data: Unused — rules operate on ``input_data`` / ``output_data`` directly.
            stage: Current stage (PRE evaluates 1-param rules; POST evaluates
                2-param rules when ``input_data`` is available, else 1-param rules).
            input_data: Normalised tool input dict.
            output_data: Normalised tool output dict (``None`` at PRE stage).

        Returns:
            ``GuardrailValidationResult`` with PASSED or VALIDATION_FAILED.
        """
        from .._base import _evaluate_rules  # decorators/_base.py

        return _evaluate_rules(self.rules, stage, input_data, output_data)
