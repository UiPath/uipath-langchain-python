import re
from typing import Any, Dict, Literal

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode

from ...exceptions import AgentTerminationException
from ..types import AgentGuardrailsGraphState
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
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GuardrailActionNode:
        raw_node_name = f"{scope.name}_{execution_stage}_{guardrail.name}_block"
        node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

        async def _node(_state: AgentGuardrailsGraphState) -> Dict[str, Any]:
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Guardrail violation",
                detail=self.reason,
                category=UiPathErrorCategory.USER,
            )

        return node_name, _node
