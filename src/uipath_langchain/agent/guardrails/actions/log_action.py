import logging
import re

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope

from uipath_langchain.agent.guardrails.types import ExecutionStage

from ..types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode


class LogAction(GuardrailAction):
    """Action that logs guardrail violation and continues."""

    def __init__(self, level: int = logging.WARNING) -> None:
        self.level = level

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
    ) -> GuardrailActionNode:
        raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_log"
        node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

        # TODO: add complete implementation for Log action
        async def _node(_state: AgentGuardrailsGraphState):
            print(
                self.level,
                "Guardrail '%s' failed at %s %s: %s",
                guardrail.name,
                execution_stage,
                scope.value if hasattr(scope, "value") else str(scope),
                _state.guardrail_validation_result,
            )
            return {}

        return node_name, _node
