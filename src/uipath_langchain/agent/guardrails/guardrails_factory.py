import logging
from typing import Sequence

from uipath.agent.models.agent import (
    AgentGuardrail,
    AgentGuardrailBlockAction,
    AgentGuardrailEscalateAction,
    AgentGuardrailLogAction,
    AgentGuardrailSeverityLevel,
    AgentUnknownGuardrail,
)
from uipath.platform.guardrails import BaseGuardrail

from uipath_langchain.agent.guardrails.actions import (
    BlockAction,
    EscalateAction,
    GuardrailAction,
    LogAction,
)


def build_guardrails_with_actions(
    guardrails: Sequence[AgentGuardrail] | None,
) -> list[tuple[BaseGuardrail, GuardrailAction]]:
    """Build a list of (guardrail, action) tuples from model definitions.

    Args:
        guardrails: Sequence of guardrail model objects or None.

    Returns:
        A list of tuples pairing each supported guardrail with its executable action.
    """
    if not guardrails:
        return []

    result: list[tuple[BaseGuardrail, GuardrailAction]] = []
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
        elif isinstance(action, AgentGuardrailEscalateAction):
            result.append(
                (
                    guardrail,
                    EscalateAction(
                        app_name=action.app.name,
                        app_folder_path=action.app.folder_name,
                        version=action.app.version,
                        assignee=action.recipient.value,
                    ),
                )
            )
    return result
