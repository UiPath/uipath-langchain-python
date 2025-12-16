import re

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode

from uipath_langchain.agent.guardrails.types import ExecutionStage

from ...exceptions import AgentTerminationException
from ...react.types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode


class BlockAction(GuardrailAction):
    """Action that terminates execution when a guardrail fails.

    Args:
        reason: Reason string to include in the raised exception title.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_block"
        node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

        async def _node(_state: AgentGuardrailsGraphState):
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Guardrail violation",
                detail=self.reason,
                category=UiPathErrorCategory.USER,
            )

        return node_name, _node
