"""Prompt injection detection guardrail validator."""

from uuid import uuid4

from uipath.core.guardrails import GuardrailScope, GuardrailSelector
from uipath.platform.guardrails import BuiltInValidatorGuardrail
from uipath.platform.guardrails.guardrails import NumberParameterValue

from ...enums import GuardrailExecutionStage
from ._base import GuardrailValidatorBase


class PromptInjectionValidator(GuardrailValidatorBase):
    """Validates LLM input for prompt injection attacks.

    Uses the UiPath prompt injection detection API. Restricted to LLM scope
    and PRE stage only — prompt injection is an input-only concern.

    Args:
        threshold: Detection confidence threshold (0.0–1.0). Default: ``0.5``.

    Raises:
        ValueError: If ``threshold`` is outside [0.0, 1.0].

    Example::

        prompt_inject = PromptInjectionValidator(threshold=0.7)

        @guardrail(
            validator=prompt_inject,
            action=BlockAction(),
            name="LLM Prompt Injection Detection",
        )
        def create_llm():
            return UiPathChat(model="gpt-4o")
    """

    supported_scopes = [GuardrailScope.LLM]
    supported_stages = [GuardrailExecutionStage.PRE]

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold

    def get_built_in_guardrail(
        self,
        scope: GuardrailScope,
        name: str,
        description: str | None,
        enabled_for_evals: bool,
    ) -> BuiltInValidatorGuardrail:
        """Build a prompt injection ``BuiltInValidatorGuardrail`` for the UiPath API.

        Args:
            scope: The resolved scope of the decorated object (must be LLM).
            name: Name for the guardrail.
            description: Optional description.
            enabled_for_evals: Whether enabled in evaluation scenarios.

        Returns:
            Configured ``BuiltInValidatorGuardrail`` for prompt injection detection.
        """
        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=name,
            description=description
            or f"Detects prompt injection with threshold {self.threshold}",
            enabled_for_evals=enabled_for_evals,
            selector=GuardrailSelector(scopes=[GuardrailScope.LLM]),
            guardrail_type="builtInValidator",
            validator_type="prompt_injection",
            validator_parameters=[
                NumberParameterValue(
                    parameter_type="number",
                    id="threshold",
                    value=self.threshold,
                ),
            ],
        )
