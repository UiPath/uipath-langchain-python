"""UiPath Guardrails middleware for LangChain agents.

This module provides a developer-friendly API for configuring guardrails
that integrate with UiPath's guardrails service.
"""

from uipath.agent.models.agent import AgentGuardrailSeverityLevel
from uipath.core.guardrails import GuardrailScope

from .actions import BlockAction, LogAction, LoggingSeverityLevel
from .decorators import (
    DeterministicValidator,
    GuardrailValidatorBase,
    PIIValidator,
    PromptInjectionValidator,
    RuleFunction,
    guardrail,
)
from .enums import GuardrailExecutionStage
from .middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
)
from .models import GuardrailAction, PIIDetectionEntity

__all__ = [
    # Decorator
    "guardrail",
    # Validators
    "GuardrailValidatorBase",
    "PIIValidator",
    "PromptInjectionValidator",
    "DeterministicValidator",
    "RuleFunction",
    # Models & enums
    "PIIDetectionEntity",
    "GuardrailExecutionStage",
    "GuardrailScope",
    "GuardrailAction",
    # Actions
    "LogAction",
    "BlockAction",
    "LoggingSeverityLevel",
    # Middlewares (unchanged)
    "UiPathPIIDetectionMiddleware",
    "UiPathPromptInjectionMiddleware",
    "UiPathDeterministicGuardrailMiddleware",
    # Re-exports
    "AgentGuardrailSeverityLevel",
]
