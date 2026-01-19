"""Example implementations of GuardrailAction.

This module provides example implementations of the GuardrailAction interface.
These are part of the SDK and demonstrate common use cases for guardrail actions.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from uipath.agent.models.agent import AgentGuardrailSeverityLevel
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.exceptions import AgentTerminationException

from .models import GuardrailAction


def _severity_to_log_level(severity: AgentGuardrailSeverityLevel) -> int:
    """Convert AgentGuardrailSeverityLevel to Python logging level."""
    mapping = {
        AgentGuardrailSeverityLevel.ERROR: logging.ERROR,
        AgentGuardrailSeverityLevel.WARNING: logging.WARNING,
        AgentGuardrailSeverityLevel.INFO: logging.INFO,
    }
    return mapping.get(severity, logging.WARNING)


@dataclass
class LogAction(GuardrailAction):
    """Example implementation: Log action for guardrails.

    This action logs guardrail violations at a specified severity level.

    Args:
        severity_level: Severity level for logging (Error, Warning, Info)
        message: Optional custom message to log. If not provided, a default
            message will be generated.
    """

    severity_level: AgentGuardrailSeverityLevel = AgentGuardrailSeverityLevel.WARNING
    message: Optional[str] = None

    def handle_validation_result(
        self,
        result: GuardrailValidationResult,
        input_data: str | dict[str, Any],
        guardrail_name: str,
    ) -> str | dict[str, Any] | None:
        """Handle validation result by logging it."""
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            log_level = _severity_to_log_level(self.severity_level)
            log_level_name = logging.getLevelName(log_level)
            message = self.message or f"Failed: {result.reason}"
            logger = logging.getLogger(__name__)
            logger.log(log_level, message)
            print(f"[{log_level_name}][GUARDRAIL] [{guardrail_name}] {message}")
        return None


@dataclass
class BlockAction(GuardrailAction):
    """Example implementation: Block action for guardrails.

    This action blocks execution by raising an AgentTerminationException when
    a guardrail validation fails.

    Args:
        title: Optional custom title for the termination exception.
            If not provided, a default title will be generated.
        detail: Optional custom detail message for the termination exception.
            If not provided, the guardrail validation reason will be used.
    """

    title: Optional[str] = None
    detail: Optional[str] = None

    def handle_validation_result(
        self,
        result: GuardrailValidationResult,
        input_data: str | dict[str, Any],
        guardrail_name: str,
    ) -> str | dict[str, Any] | None:
        """Handle validation result by blocking execution."""
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            title = self.title or f"Guardrail [{guardrail_name}] blocked execution"
            detail = self.detail or result.reason or "Guardrail validation failed"
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title=title,
                detail=detail,
            )
        return None
