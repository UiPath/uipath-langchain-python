import re
from typing import Any

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode

from uipath_langchain.agent.guardrails.types import ExecutionStage

from ...exceptions import AgentTerminationException
from ...react.types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode


class FilterAction(GuardrailAction):
    """Action that filters inputs/outputs on guardrail failure.

    For now, filtering is only supported for non-AGENT and non-LLM scopes.
    If invoked for ``GuardrailScope.AGENT`` or ``GuardrailScope.LLM``, this action
    raises an exception to indicate the operation is not supported yet.
    """

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        """Create a guardrail action node that performs filtering.

        Args:
            guardrail: The guardrail responsible for the validation.
            scope: The scope in which the guardrail applies.
            execution_stage: Whether this runs before or after execution.
            guarded_component_name: Name of the guarded component.

        Returns:
            A tuple containing the node name and the async node callable.
        """
        raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_filter"
        node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

        async def _node(_state: AgentGuardrailsGraphState) -> dict[str, Any]:
            if scope in (GuardrailScope.AGENT, GuardrailScope.LLM):
                raise AgentTerminationException(
                    code=UiPathErrorCode.EXECUTION_ERROR,
                    title="Guardrail filter action not supported",
                    detail=f"FilterAction is not supported for scope [{scope.name}] at this time.",
                    category=UiPathErrorCategory.USER,
                )
            # No-op for other scopes for now.
            return {}

        return node_name, _node
