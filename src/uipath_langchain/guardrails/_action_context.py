"""Runtime context for guardrail actions (scope / stage / component).

A :class:`GuardrailAction`'s ``handle_validation_result(result, data,
guardrail_name)`` signature does not carry the guardrail's scope, execution
stage, or guarded-component label — but the middleware that invokes the action
knows all three. The middleware publishes them here (via a ``ContextVar``) for
the duration of the action call, so actions that need them — e.g.
``EscalateAction``, which maps them onto the escalation app's ``Component`` /
``ExecutionStage`` fields — can read them at runtime instead of requiring the
developer to hardcode them.

The context is set synchronously around each action invocation and reset
afterwards, so it is correct across LangGraph's interrupt/replay too (it is
re-published on every replay).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from .enums import GuardrailExecutionStage, GuardrailScope


@dataclass(frozen=True)
class GuardrailActionContext:
    """The guardrail context active while an action handles a violation."""

    scope: GuardrailScope | None = None
    execution_stage: GuardrailExecutionStage | None = None
    component: str | None = None
    description: str | None = None
    input_payload: str | None = None


_action_context: ContextVar[GuardrailActionContext | None] = ContextVar(
    "uipath_guardrail_action_context", default=None
)


def current_action_context() -> GuardrailActionContext | None:
    """Return the guardrail context for the in-flight action call, if any."""
    return _action_context.get()


def component_label(scope: GuardrailScope | None) -> str | None:
    """Map a guardrail scope to the app's component label (matches the SDK).

    TOOL has no static label here — the tool name is supplied separately by the
    caller — so this returns ``None`` for TOOL scope.
    """
    if scope == GuardrailScope.AGENT:
        return "Agent"
    if scope == GuardrailScope.LLM:
        return "LLM call"
    return None
