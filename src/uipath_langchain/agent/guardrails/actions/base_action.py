from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Tuple

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope


class GuardrailAction(ABC):
    """Extensible action interface producing a node to enforce the action on guardrail validation failure."""

    @abstractmethod
    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: Literal["PreExecution", "PostExecution"],
    ) -> GuardrailActionNode:
        """Create and return the Action node to execute on validation failure."""
        ...


GuardrailActionNode = Tuple[str, Any]
