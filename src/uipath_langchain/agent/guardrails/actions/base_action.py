from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

from uipath.platform.guardrails import BaseGuardrail, GuardrailScope

from uipath_langchain.agent.guardrails.types import ExecutionStage

GuardrailActionNode = tuple[str, Any]

# A single node or an ordered sequence of nodes that will be chained with edges.
# When a sequence is returned the first node is the entry-point (the one the
# guardrail evaluation node routes to on failure) and the **last** node gets the
# outgoing edge to `next_node`.
GuardrailActionNodes = Union[GuardrailActionNode, Sequence[GuardrailActionNode]]


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
    ) -> GuardrailActionNodes:
        """Create and return the Action node(s) to execute on validation failure.

        May return a single ``(name, callable)`` tuple or an ordered sequence of
        such tuples.  When a sequence is returned the nodes will be chained
        together with edges in the subgraph (first → second → … → next_node).
        """
        ...
