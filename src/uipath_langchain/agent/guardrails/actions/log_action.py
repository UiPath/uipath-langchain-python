import logging
import re
from typing import Any, Optional

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope

from uipath_langchain.agent.guardrails.types import ExecutionStage

from ...react.types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode

logger = logging.getLogger(__name__)


class LogAction(GuardrailAction):
    """Action that logs guardrail violation and continues."""

    def __init__(self, message: Optional[str], level: int = logging.INFO) -> None:
        """Initialize the log action.

        Args:
            message: Message to be logged.
            level: Logging level used when reporting a guardrail failure.
        """
        self.message = message
        self.level = level

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        """Create a guardrail action node that logs validation failures.

        Args:
            guardrail: The guardrail whose failure is being logged.
            scope: The scope in which the guardrail applies.
            execution_stage: Whether this runs before or after execution.

        Returns:
            A tuple containing the node name and the async node callable.
        """
        raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_log"
        node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

        async def _node(_state: AgentGuardrailsGraphState) -> dict[str, Any]:
            message = (
                self.message
                or f"Guardrail [{guardrail.name}] validation failed for [{scope.name}] [{execution_stage.name}] with the following reason: {_state.guardrail_validation_result}"
            )

            logger.log(self.level, message)
            return {}

        return node_name, _node
