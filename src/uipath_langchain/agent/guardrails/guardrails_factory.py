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
            # TODO get app_name from name + folder_name

            result.append(
                (
                    guardrail,
                    EscalateAction(
                        app_name="app_name",
                        app_title=guardrail.action.app.title,
                        app_folder_path=guardrail.action.app.folder_name,
                        version=guardrail.action.app.version,
                        assignee=guardrail.action.recipient.value,
                    ),
                )
            )
    return result
