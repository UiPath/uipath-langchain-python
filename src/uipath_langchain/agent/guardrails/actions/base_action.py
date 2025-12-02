from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Tuple, Callable, Any

from uipath.platform.guardrails import CustomGuardrail, BuiltInValidatorGuardrail, GuardrailScope

from ..types import AgentGuardrailsGraphState


class GuardrailAction(ABC):
    """Extensible action interface producing a node for validation failure."""

    @abstractmethod
    def action_node(
        self,
        *,
        guardrail: CustomGuardrail | BuiltInValidatorGuardrail,
        scope: GuardrailScope,
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GuardrailActionNode:
        """Create and return the GraphNode to execute on validation failure."""
        ...


GuardrailActionNode = Tuple[str, Callable[[AgentGuardrailsGraphState], Any]]
