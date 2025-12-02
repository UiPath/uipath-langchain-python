from __future__ import annotations

import re
from typing import Literal, Dict, Any

from uipath.platform.guardrails import CustomGuardrail, BuiltInValidatorGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCode, UiPathErrorCategory

from ..types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode
from ...exceptions import AgentTerminationException


class BlockAction(GuardrailAction):
    """Action that terminates execution when a guardrail fails.

    Args:
        reason: Optional reason string to include in the raised exception title.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def action_node(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GuardrailActionNode:
        sanitized = re.sub(r"\W+", "_", getattr(guardrail, "name", "guardrail")).strip(
            "_"
        )
        node_name = f"{sanitized}_{execution_stage.lower()}_{scope.lower()}_block"

        async def _node(_state: AgentGuardrailsGraphState) -> Dict[str, Any]:
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="Guardrail violation",
                detail=self.reason,
                category=UiPathErrorCategory.USER,
            )

        return node_name, _node
