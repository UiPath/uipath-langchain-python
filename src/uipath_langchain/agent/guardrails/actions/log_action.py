from __future__ import annotations

import logging
import re
from typing import Literal, Dict, Any

from uipath.models.guardrails import CustomGuardrail, BuiltInValidatorGuardrail, GuardrailScope

from .base_action import GuardrailAction, GuardrailActionNode
from src.uipath_langchain.agent.guardrails.guardrail_nodes import logger
from ..types import AgentGuardrailsGraphState


class LogAction(GuardrailAction):
    """Action that logs guardrail violations and continues."""

    def __init__(self, level: int = logging.WARNING) -> None:
        self.level = level

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
        node_name = f"{sanitized}_{execution_stage.lower()}_{scope.lower()}_log"

        # TODO: add complete implementation for Log action
        async def _node(_state: AgentGuardrailsGraphState) -> Dict[str, Any]:
            logger.log(
                self.level,
                "Guardrail '%s' failed at %s %s: %s",
                guardrail.name,
                execution_stage,
                scope.value if hasattr(scope, "value") else str(scope),
                _state.guardrail_validation_result,
            )
            return {}

        return node_name, _node
