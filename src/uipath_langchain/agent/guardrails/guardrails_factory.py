from typing import List, Sequence, Tuple

from uipath.agent.models.agent import (
    AgentGuardrail,
    AgentGuardrailBlockAction,
    AgentGuardrailEscalateAction,
    AgentUnknownGuardrail,
)

from uipath_langchain.agent.guardrails.actions import (
    BlockAction,
    EscalateAction,
    GuardrailAction,
)


def build_guardrails_with_actions(
    guardrails: Sequence[AgentGuardrail] | None,
) -> List[Tuple[AgentGuardrail, GuardrailAction]]:
    if not guardrails:
        return []

    result: List[Tuple[AgentGuardrail, GuardrailAction]] = []
    for guardrail in guardrails:
        if isinstance(guardrail, AgentUnknownGuardrail):
            continue

        action = guardrail.action

        if isinstance(action, AgentGuardrailBlockAction):
            result.append((guardrail, BlockAction(action.reason)))

        if isinstance(action, AgentGuardrailEscalateAction):
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
