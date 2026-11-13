"""The ``uipath_governance.*`` decision an agent-defined guardrail check runs under.

The governance-unification contract shared with the AI Trust Layer: every
guardrail decision says where it ran (``hook``), what the evaluator decided
(``evaluator_result``), what was done (``action_applied``) and whether it was
enforced (``mode``). Agent-defined guardrails have no AI Trust Layer policy, so
their ``policy_id`` and ``policy_name`` are empty.
"""

from uipath.platform.guardrails import GuardrailGovernanceContext, GuardrailScope

from uipath_langchain.agent.guardrails.types import ExecutionStage

GOVERNANCE_VERSION = "1"

EVALUATOR_RESULT_ALLOW = "ALLOW"
EVALUATOR_RESULT_DENY = "DENY"
EVALUATOR_RESULT_ERROR = "ERROR"

ACTION_APPLIED_ALLOW = "ALLOW"
ACTION_APPLIED_DENY = "DENY"
ACTION_APPLIED_AUDIT = "AUDIT"
ACTION_APPLIED_HITL = "HITL"

MODE_ENFORCE = "ENFORCE"
MODE_SIMULATE = "SIMULATE"

_HOOKS: dict[tuple[GuardrailScope, ExecutionStage], str] = {
    (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION): "BEFORE_AGENT",
    (GuardrailScope.AGENT, ExecutionStage.POST_EXECUTION): "AFTER_AGENT",
    (GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION): "BEFORE_MODEL",
    (GuardrailScope.LLM, ExecutionStage.POST_EXECUTION): "AFTER_MODEL",
    (GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION): "TOOL_CALL",
    (GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION): "AFTER_TOOL",
}

# A Log action evaluates without enforcing: the violation is recorded and the run
# goes on, which the contract calls an AUDIT applied in SIMULATE mode. Filter
# strips what violated, so it counts as a denial.
_DECISIONS: dict[str, tuple[str, str]] = {
    "Block": (ACTION_APPLIED_DENY, MODE_ENFORCE),
    "Filter": (ACTION_APPLIED_DENY, MODE_ENFORCE),
    "Log": (ACTION_APPLIED_AUDIT, MODE_SIMULATE),
    "Escalate": (ACTION_APPLIED_HITL, MODE_ENFORCE),
}


def governance_hook(
    scope: GuardrailScope | None, execution_stage: ExecutionStage | None
) -> str | None:
    """The hook a guardrail at ``scope`` and ``execution_stage`` guards, if known."""
    try:
        key = (GuardrailScope(scope), ExecutionStage(execution_stage))
    except ValueError:
        return None
    return _HOOKS.get(key)


def governance_decision(action_type: str | None) -> tuple[str, str] | None:
    """``(action_applied, mode)`` for a violation handled by ``action_type``, if known."""
    if action_type is None:
        return None
    return _DECISIONS.get(action_type)


def governance_context(
    scope: GuardrailScope | None,
    execution_stage: ExecutionStage | None,
    action_type: str | None,
) -> GuardrailGovernanceContext | None:
    """The decision a check runs under, or ``None`` for an unknown hook or action."""
    hook = governance_hook(scope, execution_stage)
    decision = governance_decision(action_type)
    if hook is None or decision is None:
        return None
    action_on_violation, mode = decision
    return GuardrailGovernanceContext(
        hook=hook,
        mode=mode,
        version=GOVERNANCE_VERSION,
        action_on_violation=action_on_violation,
    )
