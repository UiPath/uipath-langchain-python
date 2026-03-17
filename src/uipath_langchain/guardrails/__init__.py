"""UiPath Guardrails middleware for LangChain agents.

This module provides a developer-friendly API for configuring guardrails
that integrate with UiPath's guardrails service.
"""

from uipath.agent.models.agent import AgentGuardrailSeverityLevel
from uipath.core.guardrails import GuardrailScope

from .actions import BlockAction, LogAction, LoggingSeverityLevel
from .decorators import (
    RuleFunction,
    deterministic_guardrail,
    pii_detection_guardrail,
    prompt_injection_guardrail,
)
from .enums import GuardrailExecutionStage
from .middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
)
from .models import GuardrailAction, PIIDetectionEntity

__all__ = [
    "PIIDetectionEntity",
    "GuardrailExecutionStage",
    "GuardrailScope",
    "GuardrailAction",
    "LogAction",
    "BlockAction",
    "LoggingSeverityLevel",
    "UiPathPIIDetectionMiddleware",
    "UiPathPromptInjectionMiddleware",
    "UiPathDeterministicGuardrailMiddleware",
    "pii_detection_guardrail",
    "prompt_injection_guardrail",
    "deterministic_guardrail",
    "RuleFunction",
    "AgentGuardrailSeverityLevel",  # Re-export for convenience
]
