"""Human-in-the-loop escalation action for LangChain guardrail middlewares.

``EscalateAction`` is a :class:`GuardrailAction` that, on a guardrail violation,
escalates the flagged content to a human reviewer through a UiPath **Action
App** (e.g. the *Guardrail Escalation Action App*) using the documented HITL
primitive ``interrupt(CreateEscalation(...))`` — the same mechanism coded
agents use for human-in-the-loop tasks.

It is the escalation counterpart to :class:`LogAction` and :class:`BlockAction`
for the middleware path::

    UiPathPIIDetectionMiddleware(
        scopes=[GuardrailScope.AGENT],
        action=EscalateAction(
            app_name="Guardrail.Escalation.Action.App.2",
            app_folder_path="Shared",
        ),
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
    )

Lifecycle:

1. On ``VALIDATION_FAILED``, the flagged content is mapped onto the action
   app's input schema and ``interrupt(CreateEscalation(...))`` is raised. The
   platform creates the task, suspends the run durably, and resumes it once a
   human acts. ``interrupt()`` is memoized, so replay-on-resume never creates a
   duplicate task.
2. On resume, the completed task's outcome drives the result:
   - ``Approve`` → return the reviewer-edited content so the middleware
     substitutes it, or keep the original when the reviewed value is
     absent/empty (matching the factory-path ``EscalateAction``). The reviewed
     value is read from ``ReviewedInputs`` for a PRE (input) escalation and from
     ``ReviewedOutputs`` for a POST (output) one.
   - ``Reject`` → raise :class:`GuardrailBlockException`, terminating the run
     (mirroring the SDK's factory-path ``EscalateAction``).

The flagged content is mapped onto the action app's schema using the guardrail
context the middleware publishes (scope / stage / component / description): a PRE
violation fills ``Inputs``; a POST violation fills ``Outputs`` and carries the
original input in ``Inputs``, so the reviewer sees both.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langgraph.types import interrupt
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.action_center.tasks import TaskRecipient
from uipath.platform.common import CreateEscalation, UiPathConfig
from uipath.platform.guardrails.decorators import (
    GuardrailAction,
    GuardrailBlockException,
)
from uipath.platform.hitl import HitlSchema

from ._action_context import GuardrailActionContext, current_action_context
from .enums import GuardrailExecutionStage

logger = logging.getLogger(__name__)


@dataclass
class EscalateAction(GuardrailAction):
    """Escalate guardrail violations to a human via a UiPath Action App.

    The escalation-app payload fields — ``Component``, ``ExecutionStage``,
    ``GuardrailDescription``, and the ``Inputs`` / ``Outputs`` input-vs-output
    split — are derived at runtime from the guardrail context the middleware
    publishes (scope → component, hook → stage, guardrail → description), so they
    are not configured here.

    Args:
        app_name: Name of the published escalation Action App.
        app_folder_path: Folder where the app is deployed.
        assignee: Optional task assignee — the simple username/email shortcut.
        recipient: Optional typed escalation target (``TaskRecipient``). Use this
            to target a ``UserId`` / ``GroupId`` / ``UserEmail`` / ``GroupName``
            (the recipient types the HITL ``CreateEscalation`` primitive
            supports). Takes precedence over ``assignee`` when set.
        title: Optional task title. Defaults to a message derived from the
            guardrail name.
    """

    app_name: str
    app_folder_path: str | None = None
    assignee: str | None = None
    recipient: TaskRecipient | None = None
    title: str | None = None

    def handle_validation_result(
        self,
        result: GuardrailValidationResult,
        data: str | dict[str, Any],
        guardrail_name: str,
    ) -> str | dict[str, Any] | None:
        """Escalate to the action app and apply the reviewer's decision."""
        if result.result != GuardrailValidationResultType.VALIDATION_FAILED:
            return None

        ctx = current_action_context()
        data_is_dict = isinstance(data, dict)
        # JSON-encode the flagged payload. The action app parses this field as
        # JSON (the helix backend sends it via ToJsonString and the factory-path
        # EscalateAction via json.dumps), so a raw string leaves the app's
        # input box empty.
        content = json.dumps(data)

        logger.warning(
            "[GUARDRAIL] [%s] violation detected — escalating to app '%s'.",
            guardrail_name,
            self.app_name,
        )

        raw = interrupt(
            CreateEscalation(
                app_name=self.app_name,
                app_folder_path=self.app_folder_path,
                title=self.title or f"Guardrail '{guardrail_name}': review required",
                data=self._build_app_inputs(guardrail_name, result, content, ctx),
                assignee=self.assignee,
                recipient=self.recipient,
            )
        )

        outcome, response = _normalize_escalation_result(raw)
        logger.info(
            "[GUARDRAIL] [%s] escalation resolved with outcome '%s'.",
            guardrail_name,
            outcome,
        )

        if outcome.lower() == "approve":
            reviewed = response.get(_reviewed_field_name(ctx))
            if not reviewed:
                return None
            return _coerce_reviewed(reviewed, data_is_dict)

        reason = response.get("Reason") or result.reason or "No reason provided."
        raise GuardrailBlockException(
            title=f"Guardrail [{guardrail_name}] escalation rejected",
            detail=reason,
        )

    def _build_app_inputs(
        self,
        guardrail_name: str,
        result: GuardrailValidationResult,
        content: str,
        ctx: GuardrailActionContext | None,
    ) -> dict[str, Any]:
        """Map the guardrail context onto the action app's input schema.

        Keys mirror the deployed ``Guardrail Escalation Action App`` schema. The
        flagged content uses ``Inputs`` / ``Outputs``, split by the published
        execution stage:

        - PRE  → flagged content fills ``Inputs`` (``Outputs`` empty).
        - POST → flagged content fills ``Outputs``; the original input
          (``ctx.input_payload``) fills ``Inputs`` so the reviewer sees both.
        """
        is_post = bool(ctx and ctx.execution_stage == GuardrailExecutionStage.POST)
        data: dict[str, Any] = {
            "GuardrailName": guardrail_name,
            "GuardrailDescription": (ctx.description if ctx else None) or "",
            "GuardrailResult": result.reason or "",
        }
        if is_post:
            data["Inputs"] = (ctx.input_payload if ctx else None) or ""
            data["Outputs"] = content
        else:
            data["Inputs"] = content
            data["Outputs"] = ""
        # Legacy aliases kept for older apps that read the Tool-prefixed fields.
        data["ToolInputs"] = data["Inputs"]
        data["ToolOutputs"] = data["Outputs"]
        if ctx and ctx.component:
            data["Component"] = ctx.component
            data["Tool"] = ctx.component
        execution_stage = _execution_stage_label(ctx.execution_stage if ctx else None)
        if execution_stage:
            data["ExecutionStage"] = execution_stage
        tenant_name = _resolve_tenant_name()
        if tenant_name:
            data["TenantName"] = tenant_name
        trace_url = _agent_trace_url()
        if trace_url:
            data["AgentTrace"] = trace_url
        return data


def _normalize_escalation_result(raw: Any) -> tuple[str, dict[str, Any]]:
    """Normalize the interrupt resume value into ``(outcome, data)``.

    ``CreateEscalation`` resumes with the completed task. Depending on how the
    platform delivers it, ``raw`` may be a task-like object (``.action`` /
    ``.data``) or a plain dict, so both are handled.
    """
    if raw is None:
        return "Approve", {}

    if isinstance(raw, dict):
        outcome = raw.get("action") or raw.get("Action") or raw.get("outcome")
        data = raw.get("data")
        if not isinstance(data, dict):
            data = raw
        return (str(outcome) if outcome else "Approve"), data

    outcome = getattr(raw, "action", None)
    data = getattr(raw, "data", None)
    if not isinstance(data, dict):
        data = {}
    return (str(outcome) if outcome else "Approve"), data


def _coerce_reviewed(reviewed: Any, want_dict: bool) -> str | dict[str, Any]:
    """Coerce the reviewed value back to the original data's shape."""
    if want_dict and isinstance(reviewed, str):
        try:
            parsed = json.loads(reviewed)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return reviewed


def _execution_stage_label(stage: GuardrailExecutionStage | None) -> str | None:
    """Map a guardrail execution stage to the action app's stage label."""
    if stage == GuardrailExecutionStage.PRE:
        return "PreExecution"
    if stage == GuardrailExecutionStage.POST:
        return "PostExecution"
    return None


def _reviewed_field_name(ctx: GuardrailActionContext | None) -> str:
    """Return the resume field the reviewer edits, keyed by execution stage.

    A PRE (input) escalation comes back as ``ReviewedInputs``; a POST (output)
    one as ``ReviewedOutputs`` — matching the action app's output schema. With no
    stage context we default to ``ReviewedInputs``.
    """
    stage = ctx.execution_stage if ctx else None
    if stage == GuardrailExecutionStage.POST:
        return "ReviewedOutputs"
    return "ReviewedInputs"


def _safe_config_attr(attr: str) -> Any:
    """Read an attribute off ``UiPathConfig``, tolerating missing context."""
    try:
        return getattr(UiPathConfig, attr, None)
    except Exception:  # pragma: no cover - config not always populated locally
        return None


def _resolve_tenant_name() -> str | None:
    """Resolve the tenant name for the escalation payload.

    Prefers ``UiPathConfig.tenant_name`` (the ``UIPATH_TENANT_NAME`` env var,
    injected by Orchestrator in deployed runs). Falls back to parsing the
    tenant segment from the base URL — which ``uipath auth`` writes as
    ``UIPATH_URL`` (e.g. ``https://.../<org>/<tenant>``) — so the logged-in
    tenant is populated for local ``uipath run`` too.
    """
    name = _safe_config_attr("tenant_name")
    if name:
        return name
    base_url = _safe_config_attr("base_url")
    if not base_url:
        return None
    try:
        from uipath._utils import UiPathUrl

        return UiPathUrl(base_url).tenant_name or None
    except Exception:  # pragma: no cover - defensive
        return None


def _agent_trace_url() -> str | None:
    """Build the agent execution viewer URL from ``UiPathConfig``.

    Mirrors the factory-path ``EscalateAction`` so the action app's "Agent
    trace" field links to the run. Returns ``None`` when the runtime context
    (base URL / trace identifiers) isn't populated — e.g. local runs without a
    deployment context — so we never emit a URL containing ``None`` segments.
    """
    base_url = _safe_config_attr("base_url")
    organization_id = _safe_config_attr("organization_id")
    if not base_url or not organization_id:
        return None
    try:
        from uipath._utils import UiPathUrl

        normalized = UiPathUrl(base_url).base_url
        if _safe_config_attr("is_studio_project"):
            project_id = _safe_config_attr("project_id")
            solution_id = _safe_config_attr("studio_solution_id")
            if not project_id:
                return None
            return (
                f"{normalized}/{organization_id}/studio_/designer/"
                f"{project_id}?solutionId={solution_id}"
            )
        folder_key = _safe_config_attr("folder_key")
        process_uuid = _safe_config_attr("process_uuid")
        trace_id = _safe_config_attr("trace_id")
        project_key = _safe_config_attr("project_key")
        package_version = _safe_config_attr("process_version")
        if not (folder_key and process_uuid and trace_id):
            return None
        return (
            f"{normalized}/{organization_id}/agents_/deployed/{folder_key}/"
            f"{process_uuid}/{project_key}/{package_version}/traces/{trace_id}"
        )
    except Exception:  # pragma: no cover - defensive: never break escalation
        return None
