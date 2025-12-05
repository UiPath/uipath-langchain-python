import logging
from typing import Sequence

from uipath.agent.models.agent import (
    AgentGuardrail,
    AgentGuardrailBlockAction,
    AgentGuardrailLogAction,
    AgentGuardrailSeverityLevel,
    AgentUnknownGuardrail,
)

from uipath_langchain.agent.guardrails.actions import (
    BlockAction,
    GuardrailAction,
    LogAction,
)


def build_guardrails_with_actions(
    guardrails: Sequence[AgentGuardrail] | None,
) -> list[tuple[AgentGuardrail, GuardrailAction]]:
    """Build a list of (guardrail, action) tuples from model definitions.

    Args:
        guardrails: Sequence of guardrail model objects or None.

    Returns:
        A list of tuples pairing each supported guardrail with its executable action.
    """
    if not guardrails:
        return []

    result: list[tuple[AgentGuardrail, GuardrailAction]] = []
    for guardrail in guardrails:
        if isinstance(guardrail, AgentUnknownGuardrail):
            continue

        action = guardrail.action

        if isinstance(action, AgentGuardrailBlockAction):
            result.append((guardrail, BlockAction(action.reason)))
        elif isinstance(action, AgentGuardrailLogAction):
            severity_level_map = {
                AgentGuardrailSeverityLevel.ERROR: logging.ERROR,
                AgentGuardrailSeverityLevel.WARNING: logging.WARNING,
                AgentGuardrailSeverityLevel.INFO: logging.INFO,
            }
            level = severity_level_map.get(action.severity_level, logging.INFO)
            result.append(
                (
                    guardrail,
                    LogAction(message=action.message, level=level),
                )
            )
    return result
