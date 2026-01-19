"""UiPath Guardrails middleware for LangChain agents.

This module provides a developer-friendly API for configuring guardrails
that integrate with UiPath's guardrails service.
"""

from uipath.agent.models.agent import AgentGuardrailSeverityLevel
from uipath.core.guardrails import GuardrailScope

from .actions import BlockAction, LogAction
from .enums import GuardrailExecutionStage, PIIDetectionEntityType
from .middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
)
from .models import GuardrailAction, PIIDetectionEntity

__all__ = [
    "PIIDetectionEntityType",
    "GuardrailExecutionStage",
    "GuardrailScope",
    "PIIDetectionEntity",
    "GuardrailAction",
    "LogAction",
    "BlockAction",
    "UiPathPIIDetectionMiddleware",
    "UiPathPromptInjectionMiddleware",
    "UiPathDeterministicGuardrailMiddleware",
    "AgentGuardrailSeverityLevel",  # Re-export for convenience
]
