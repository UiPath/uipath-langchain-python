"""Abstract base class for guardrail validators."""

from typing import Any, ClassVar

from uipath.core.guardrails import GuardrailScope, GuardrailValidationResult
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from ...enums import GuardrailExecutionStage  # guardrails/enums.py


class GuardrailValidatorBase:
    """Abstract base class for guardrail validators.

    Defines WHAT to validate. The @guardrail decorator defines HOW to respond
    (action, name, stage, enabled_for_evals).

    A validator instance can be declared once and reused across multiple
    @guardrail decorators with different actions, names, or stages.

    Subclasses implement either:
    - ``get_built_in_guardrail()`` for UiPath API-based validation (PII, PromptInjection)
    - ``evaluate()`` for local Python-based validation (Deterministic)
    """

    supported_scopes: ClassVar[list[GuardrailScope]] = []
    """Scopes this validator supports. Empty list means all scopes are allowed."""

    supported_stages: ClassVar[list[GuardrailExecutionStage]] = []
    """Stages this validator supports. Empty list means all stages are allowed."""

    def get_built_in_guardrail(
        self,
        scope: GuardrailScope,
        name: str,
        description: str | None,
        enabled_for_evals: bool,
    ) -> BuiltInValidatorGuardrail | None:
        """Build a UiPath API guardrail instance for this validator.

        API-based validators (PII, PromptInjection) override this to return a
        ``BuiltInValidatorGuardrail``. Local validators return ``None`` (default).

        Args:
            scope: The resolved scope of the decorated object.
            name: Name for the guardrail.
            description: Optional description.
            enabled_for_evals: Whether enabled in evaluation scenarios.

        Returns:
            ``BuiltInValidatorGuardrail`` for API-based evaluation, or ``None``
            to use local ``evaluate()`` instead.
        """
        return None

    def evaluate(
        self,
        data: str | dict[str, Any],
        stage: GuardrailExecutionStage,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
    ) -> GuardrailValidationResult:
        """Perform local validation (no UiPath API call).

        Local validators (Deterministic) override this. Only called when
        ``get_built_in_guardrail()`` returns ``None``.

        Args:
            data: The primary data being evaluated (message text, tool I/O dict).
            stage: Current execution stage (PRE or POST).
            input_data: Normalised tool/agent input dict, or ``None`` if unavailable.
            output_data: Normalised tool/agent output dict, or ``None`` at PRE stage.

        Returns:
            ``GuardrailValidationResult`` with PASSED or VALIDATION_FAILED.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement either get_built_in_guardrail() "
            "for API-based validation or evaluate() for local validation."
        )

    def validate_scope(self, scope: GuardrailScope) -> None:
        """Raise ``ValueError`` if ``scope`` is not supported by this validator.

        Args:
            scope: The resolved scope of the decorated object.

        Raises:
            ValueError: If ``supported_scopes`` is non-empty and ``scope`` is absent.
        """
        if self.supported_scopes and scope not in self.supported_scopes:
            raise ValueError(
                f"{type(self).__name__} does not support scope {scope!r}. "
                f"Supported scopes: {[s.value for s in self.supported_scopes]}"
            )

    def validate_stage(self, stage: GuardrailExecutionStage) -> None:
        """Raise ``ValueError`` if ``stage`` is not supported by this validator.

        Args:
            stage: The requested execution stage.

        Raises:
            ValueError: If ``supported_stages`` is non-empty and ``stage`` is absent.
        """
        if self.supported_stages and stage not in self.supported_stages:
            raise ValueError(
                f"{type(self).__name__} does not support stage {stage!r}. "
                f"Supported stages: {[s.value for s in self.supported_stages]}"
            )
