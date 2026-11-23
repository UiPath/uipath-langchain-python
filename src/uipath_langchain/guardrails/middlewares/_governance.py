"""The ``uipath_governance.*`` attributes of a guardrail check's spans.

Where the check runs (``hook``), what it decided and what the agent did about
it. No AI Trust Layer policy enforces a guardrail an agent defines, so its
policy keys are empty.
"""

from dataclasses import dataclass
from typing import Any

from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails.decorators import GuardrailExecutionStage

VERSION = "1"

_PREFIX = "uipath_governance."

_HOOKS: dict[tuple[GuardrailScope, GuardrailExecutionStage], str] = {
    (GuardrailScope.AGENT, GuardrailExecutionStage.PRE): "BEFORE_AGENT",
    (GuardrailScope.AGENT, GuardrailExecutionStage.POST): "AFTER_AGENT",
    (GuardrailScope.LLM, GuardrailExecutionStage.PRE): "BEFORE_MODEL",
    (GuardrailScope.LLM, GuardrailExecutionStage.POST): "AFTER_MODEL",
    (GuardrailScope.TOOL, GuardrailExecutionStage.PRE): "TOOL_CALL",
    (GuardrailScope.TOOL, GuardrailExecutionStage.POST): "AFTER_TOOL",
}

# ``execution_stage`` values of ``evaluate_guardrail``.
_EXECUTION_STAGES = {
    GuardrailExecutionStage.PRE: "preExecution",
    GuardrailExecutionStage.POST: "postExecution",
}

# Action type -> (action applied on a violation, mode). Log records the
# violation and lets the run go on: an AUDIT in SIMULATE mode. Filter strips
# what violated, a denial.
_DECISIONS: dict[str, tuple[str, str]] = {
    "Block": ("DENY", "ENFORCE"),
    "Filter": ("DENY", "ENFORCE"),
    "Log": ("AUDIT", "SIMULATE"),
    "Escalate": ("HITL", "ENFORCE"),
}


def hook(
    scope: GuardrailScope | None, stage: GuardrailExecutionStage | None
) -> str | None:
    """The hook a check at ``scope`` and ``stage`` guards, if known."""
    return _HOOKS.get((scope, stage))  # type: ignore[arg-type]


def action_type(action: Any) -> str | None:
    """``BlockAction`` -> ``Block``; ``None`` for an action outside the contract."""
    name = type(action).__name__.removesuffix("Action")
    return name if name in _DECISIONS else None


def container_attributes(
    scope: GuardrailScope, stage: GuardrailExecutionStage
) -> dict[str, str]:
    """Mark a stage container as governance and say which hook it guards."""
    stage_hook = hook(scope, stage)
    if stage_hook is None:
        return {}
    return {f"{_PREFIX}version": VERSION, f"{_PREFIX}hook": stage_hook}


@dataclass(frozen=True)
class GovernanceContext:
    """Where one guardrail check runs and what it does on a violation."""

    scope: GuardrailScope | None
    stage: GuardrailExecutionStage | None
    action: Any
    guardrail_name: str
    guardrail_type: str

    def evaluation_options(self) -> dict[str, Any]:
        """The ``scope``, ``execution_stage`` and ``action_type`` of ``evaluate_guardrail``.

        Empty unless all three are known.
        """
        stage = _EXECUTION_STAGES.get(self.stage)  # type: ignore[arg-type]
        kind = action_type(self.action)
        if self.scope is None or stage is None or kind is None:
            return {}
        return {"scope": self.scope, "execution_stage": stage, "action_type": kind}

    def attributes(self, result: GuardrailValidationResult) -> dict[str, Any]:
        """The check's decision on ``result``; empty if the hook or action is unknown."""
        check_hook = hook(self.scope, self.stage)
        kind = action_type(self.action)
        if check_hook is None or kind is None:
            return {}
        action_on_violation, mode = _DECISIONS[kind]
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            # A check that could not evaluate fails closed: the action still runs.
            evaluator_result = "ERROR" if result.error_type else "DENY"
            action_applied = action_on_violation
            evidence = result.reason or None
        else:
            evaluator_result = action_applied = "ALLOW"
            evidence = result.reason or f"{self.guardrail_name} passed"
        attributes = {
            "policy_id": "",
            "policy_name": "",
            "hook": check_hook,
            "evaluator_result": evaluator_result,
            "action_applied": action_applied,
            "mode": mode,
            "version": VERSION,
            "evidence": evidence,
            "guardrail_type": self.guardrail_type,
            "guardrail_name": self.guardrail_name,
        }
        return {
            f"{_PREFIX}{key}": value
            for key, value in attributes.items()
            if value is not None
        }
