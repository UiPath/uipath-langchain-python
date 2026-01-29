from abc import ABC, abstractmethod
from typing import Any

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope

from uipath_langchain.agent.guardrails.types import ExecutionStage

GuardrailActionNode = tuple[str, Any]


class GuardrailAction(ABC):
    """Extensible action interface producing a node to enforce the action on guardrail validation failure."""

    @property
    @abstractmethod
    def action_type(self) -> str:
        """Return the action type identifier (e.g., 'Block', 'Log', 'Filter', 'Escalate')."""
        ...

    @abstractmethod
    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        """Create and return the Action node to execute on validation failure."""
        ...
