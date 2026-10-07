import re

from uipath.platform.guardrails import (
    BaseGuardrail,
    GuardrailScope,
    GuardrailTerminationMode,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.guardrails.types import ExecutionStage

from ...exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ...react.types import AgentGuardrailsGraphState
from .base_action import GuardrailAction, GuardrailActionNode


def _flagged_files_note(state: AgentGuardrailsGraphState) -> str:
    names = [
        printable
        for name in state.inner_state.guardrail_flagged_file_names or []
        if (printable := _printable(name))
    ]
    return f"\nFlagged files: {', '.join(names)}" if names else ""


def _printable(name: str) -> str:
    # File names are user-supplied; a line break or bidi control would forge lines in the error.
    return "".join(ch if ch.isprintable() else " " for ch in name).strip()


class BlockAction(GuardrailAction):
    """Action that terminates execution when a guardrail fails.

    Args:
        reason: Reason string to include in the raised exception title.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def action_type(self) -> str:
        return "Block"

    @property
    def termination_mode(self) -> GuardrailTerminationMode:
        """The run stops at the first violation, so scanning can too."""
        return GuardrailTerminationMode.FAIL_FAST

    def action_node(
        self,
        *,
        guardrail: BaseGuardrail,
        scope: GuardrailScope,
        execution_stage: ExecutionStage,
        guarded_component_name: str,
    ) -> GuardrailActionNode:
        raw_node_name = f"{scope.name}_{execution_stage.name}_{guardrail.name}_block"
        node_name = re.sub(r"\W+", "_", raw_node_name.lower()).strip("_")

        async def _node(_state: AgentGuardrailsGraphState):
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION,
                title="Guardrail violation",
                detail=f"Execution was blocked by guardrail [{guardrail.name}], with reason: {self.reason}{_flagged_files_note(_state)}",
                category=UiPathErrorCategory.USER,
            )

        _node.__metadata__ = {  # type: ignore[attr-defined]
            "reason": self.reason,
            "guardrail": guardrail,
            "scope": scope,
            "execution_stage": execution_stage,
        }

        return node_name, _node
