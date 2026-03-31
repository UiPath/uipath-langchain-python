"""Custom (rule-based) guardrail validator."""

import inspect
from typing import Any, Callable, ClassVar

from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)

from ...enums import GuardrailExecutionStage
from ._base import GuardrailValidatorBase

RuleFunction = (
    Callable[[dict[str, Any]], bool] | Callable[[dict[str, Any], dict[str, Any]], bool]
)
"""Type alias for custom rule functions.

A rule is a callable that returns ``True`` to signal a violation, ``False``
to pass. It accepts either:

- One parameter: ``(args_dict) -> bool`` — evaluated at PRE or POST stage using
  the input or output dict respectively.
- Two parameters: ``(input_dict, output_dict) -> bool`` — evaluated at POST stage
  only, with access to both dicts.

Use ``and`` / ``or`` to combine multiple checks inline::

    rule=lambda args: "donkey" in args.get("joke", "").lower() or len(args.get("joke", "")) > 500
"""


class CustomValidator(GuardrailValidatorBase):
    """Validates tool input/output using a local Python rule function.

    No UiPath API call is made. The rule runs in-process, making this suitable
    for fast, custom checks such as keyword filtering, length limits, regex
    matching, or any combination using ``and`` / ``or``.

    Restricted to TOOL scope only.

    Args:
        rule: A callable that returns ``True`` to flag a violation, ``False``
            to pass. Accepts one argument (the input or output dict) or two
            arguments (input dict, output dict — POST stage only). Use
            ``and`` / ``or`` operators to combine multiple checks inline.

    Raises:
        ValueError: If ``rule`` is not callable or has an unsupported parameter
            count (must be 1 or 2).

    Example::

        @guardrail(
            validator=CustomValidator(lambda args: "donkey" in args.get("joke", "").lower()),
            action=CustomFilterAction(word_to_filter="donkey", replacement="[censored]"),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Word Filter",
        )
        @tool
        def analyze_joke_syntax(joke: str) -> str: ...

        # Always-apply action (unconditional):
        @guardrail(
            validator=CustomValidator(lambda args: True),
            action=CustomFilterAction(word_to_filter="words", replacement="words++"),
            stage=GuardrailExecutionStage.POST,
            name="Joke Content Always Filter",
        )
        @tool
        def analyze_joke_syntax(joke: str) -> str: ...
    """

    supported_scopes = [GuardrailScope.TOOL]
    supported_stages: ClassVar[list[GuardrailExecutionStage]] = []  # all stages allowed

    def __init__(self, rule: RuleFunction) -> None:
        if not callable(rule):
            raise ValueError(f"rule must be callable, got {type(rule)}")
        sig = inspect.signature(rule)
        param_count = len(sig.parameters)
        if param_count not in (1, 2):
            raise ValueError(
                f"rule must have 1 or 2 parameters, got {param_count}"
            )
        self.rule = rule
        self._param_count = param_count

    def evaluate(
        self,
        data: str | dict[str, Any],
        stage: GuardrailExecutionStage,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
    ) -> GuardrailValidationResult:
        """Evaluate the rule locally against the tool input or output dict.

        Args:
            data: Unused — the rule operates on ``input_data`` / ``output_data``.
            stage: Current stage (PRE or POST).
            input_data: Normalised tool input dict.
            output_data: Normalised tool output dict (``None`` at PRE stage).

        Returns:
            ``GuardrailValidationResult`` with PASSED or VALIDATION_FAILED.
        """
        try:
            if self._param_count == 2:
                # Two-parameter rules require both dicts — POST stage only.
                if input_data is None or output_data is None:
                    return GuardrailValidationResult(
                        result=GuardrailValidationResultType.PASSED,
                        reason="Two-parameter rule skipped: input or output data unavailable",
                    )
                violation = self.rule(input_data, output_data)  # type: ignore[call-arg]
            else:
                # One-parameter rules use input_data at PRE, output_data at POST.
                target = (
                    input_data if stage == GuardrailExecutionStage.PRE else output_data
                )
                if target is None:
                    return GuardrailValidationResult(
                        result=GuardrailValidationResultType.PASSED,
                        reason="Rule skipped: data unavailable at this stage",
                    )
                violation = self.rule(target)  # type: ignore[call-arg]
        except Exception as exc:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED,
                reason=f"Rule raised exception: {exc}",
            )

        if violation:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.VALIDATION_FAILED,
                reason="Rule detected violation",
            )
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED,
            reason="Rule passed",
        )
