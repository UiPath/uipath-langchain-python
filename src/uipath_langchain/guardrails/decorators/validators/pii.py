"""PII detection guardrail validator."""

from typing import Any, Sequence
from uuid import uuid4

from uipath.core.guardrails import GuardrailScope, GuardrailSelector
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    EnumListParameterValue,
    MapEnumParameterValue,
)

from ...models import PIIDetectionEntity
from ._base import GuardrailValidatorBase


class PIIValidator(GuardrailValidatorBase):
    """Validates data for PII (Personally Identifiable Information) entities.

    Uses the UiPath PII detection API to identify entities such as email addresses,
    phone numbers, credit card numbers, and other PII types.

    Supported at all scopes (AGENT, LLM, TOOL) and all stages.

    A single ``PIIValidator`` instance can be declared once and reused across
    multiple ``@guardrail`` decorators with different actions or stages.

    Args:
        entities: One or more ``PIIDetectionEntity`` objects specifying which PII
            types to detect and their confidence thresholds (0.0–1.0).

    Raises:
        ValueError: If ``entities`` is empty.

    Example::

        pii_email = PIIValidator(
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)]
        )

        @guardrail(
            validator=pii_email,
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            name="LLM PII Detection",
            stage=GuardrailExecutionStage.PRE,
        )
        def create_llm():
            return UiPathChat(model="gpt-4o")

        @guardrail(
            validator=pii_email,
            action=BlockAction(),
            name="Tool PII Detection",
        )
        @tool
        def my_tool(text: str) -> str: ...
    """

    # All scopes and stages supported — inherits empty lists from base (unrestricted).

    def __init__(self, entities: Sequence[PIIDetectionEntity]) -> None:
        if not entities:
            raise ValueError("entities must be provided and non-empty")
        self.entities = list(entities)

    def build_built_in_guardrail(
        self,
        scope: GuardrailScope,
        name: str,
        description: str | None,
        enabled_for_evals: bool,
    ) -> BuiltInValidatorGuardrail:
        """Build a PII detection ``BuiltInValidatorGuardrail`` for the UiPath API.

        Args:
            scope: The resolved scope of the decorated object.
            name: Name for the guardrail.
            description: Optional description.
            enabled_for_evals: Whether enabled in evaluation scenarios.

        Returns:
            Configured ``BuiltInValidatorGuardrail`` for PII detection.
        """
        entity_names = [entity.name for entity in self.entities]
        entity_thresholds: dict[str, Any] = {
            entity.name: entity.threshold for entity in self.entities
        }

        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=name,
            description=description
            or f"Detects PII entities: {', '.join(entity_names)}",
            enabled_for_evals=enabled_for_evals,
            selector=GuardrailSelector(scopes=[scope]),
            guardrail_type="builtInValidator",
            validator_type="pii_detection",
            validator_parameters=[
                EnumListParameterValue(
                    parameter_type="enum-list",
                    id="entities",
                    value=entity_names,
                ),
                MapEnumParameterValue(
                    parameter_type="map-enum",
                    id="entityThresholds",
                    value=entity_thresholds,
                ),
            ],
        )
