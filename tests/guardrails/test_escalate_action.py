"""Unit tests for the middleware ``EscalateAction``.

Drives ``EscalateAction.handle_validation_result(result, data, name)`` directly
(the guardrail-middleware action contract), patching:
- ``escalate_action.interrupt`` to simulate the resume value (Approve/Reject), and
- ``escalate_action.UiPathConfig`` to control TenantName/AgentTrace derivation.

The guardrail runtime context (scope/stage/component) that the middleware would
publish is set explicitly via ``_action_context`` where a case needs it.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath.platform.common import CreateEscalation
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

import uipath_langchain.guardrails.escalate_action as escalate_module
from uipath_langchain.guardrails import EscalateAction, GuardrailExecutionStage
from uipath_langchain.guardrails._action_context import (
    GuardrailActionContext,
    _action_context,
)
from uipath_langchain.guardrails.escalate_action import (
    _coerce_reviewed,
    _execution_stage_label,
    _normalize_escalation_result,
    _resolve_tenant_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _failed(reason: str = "PII detected: Email") -> GuardrailValidationResult:
    return GuardrailValidationResult(
        result=GuardrailValidationResultType.VALIDATION_FAILED, reason=reason
    )


def _passed() -> GuardrailValidationResult:
    return GuardrailValidationResult(
        result=GuardrailValidationResultType.PASSED, reason=""
    )


def _action(**kwargs: Any) -> EscalateAction:
    kwargs.setdefault("app_name", "Guardrail.Escalation.Action.App.2")
    kwargs.setdefault("app_folder_path", "Shared")
    return EscalateAction(**kwargs)


@contextmanager
def _published_context(
    *,
    scope: GuardrailScope | None = None,
    stage: GuardrailExecutionStage | None = None,
    component: str | None = None,
    description: str | None = None,
    input_payload: str | None = None,
) -> Iterator[None]:
    """Publish a guardrail action context for the duration of the block."""
    token = _action_context.set(
        GuardrailActionContext(
            scope=scope,
            execution_stage=stage,
            component=component,
            description=description,
            input_payload=input_payload,
        )
    )
    try:
        yield
    finally:
        _action_context.reset(token)


@contextmanager
def _patched(resume: Any, config: SimpleNamespace | None = None) -> Iterator[Any]:
    """Patch interrupt (to return ``resume``) and UiPathConfig; yields the interrupt mock."""
    cfg = (
        config
        if config is not None
        else SimpleNamespace(tenant_name=None, base_url=None)
    )
    with (
        patch.object(
            escalate_module, "interrupt", return_value=resume
        ) as mock_interrupt,
        patch.object(escalate_module, "UiPathConfig", cfg),
    ):
        yield mock_interrupt


# ---------------------------------------------------------------------------
# Triggering: PASSED vs VALIDATION_FAILED
# ---------------------------------------------------------------------------


class TestTriggering:
    def test_passed_returns_none_without_interrupt(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            result = _action().handle_validation_result(_passed(), "data", "g")
        assert result is None
        mock_interrupt.assert_not_called()

    def test_failed_calls_interrupt_with_create_escalation(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            _action(assignee="user@x.com").handle_validation_result(
                _failed("PII detected: Email"), "the topic", "PII guard"
            )
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert isinstance(payload, CreateEscalation)
        assert payload.app_name == "Guardrail.Escalation.Action.App.2"
        assert payload.app_folder_path == "Shared"
        assert payload.assignee == "user@x.com"
        assert payload.title == "Guardrail 'PII guard': review required"
        assert payload.data is not None
        assert payload.data["GuardrailName"] == "PII guard"
        assert payload.data["GuardrailDescription"] == ""
        assert payload.data["GuardrailResult"] == "PII detected: Email"


# ---------------------------------------------------------------------------
# Payload: JSON-encoding of the flagged content
# ---------------------------------------------------------------------------


class TestPayloadEncoding:
    def test_string_payload_is_json_encoded(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            _action().handle_validation_result(_failed(), "topic with email", "g")
        data = mock_interrupt.call_args[0][0].data
        assert data["Inputs"] == json.dumps("topic with email")
        assert data["Outputs"] == ""
        assert data["ToolInputs"] == data["Inputs"]
        assert data["ToolOutputs"] == data["Outputs"]

    def test_dict_payload_is_json_object(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            _action().handle_validation_result(_failed(), {"email": "a@b.com"}, "g")
        data = mock_interrupt.call_args[0][0].data
        assert json.loads(data["Inputs"]) == {"email": "a@b.com"}
        assert data["ToolInputs"] == data["Inputs"]


# ---------------------------------------------------------------------------
# Payload: Component / ExecutionStage from runtime context
# ---------------------------------------------------------------------------


class TestContextDerivedFields:
    def test_agent_pre_context(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            with _published_context(
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.PRE,
                component="Agent",
            ):
                _action().handle_validation_result(_failed(), "x", "g")
        data = mock_interrupt.call_args[0][0].data
        assert data["Component"] == "Agent"
        assert data["Tool"] == "Agent"
        assert data["ExecutionStage"] == "PreExecution"

    def test_tool_post_context(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            with _published_context(
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.POST,
                component="my_tool",
            ):
                _action().handle_validation_result(_failed(), {"a": 1}, "g")
        data = mock_interrupt.call_args[0][0].data
        assert data["Component"] == "my_tool"
        assert data["Tool"] == "my_tool"
        assert data["ExecutionStage"] == "PostExecution"

    def test_no_context_omits_component_and_stage(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            _action().handle_validation_result(_failed(), "x", "g")
        data = mock_interrupt.call_args[0][0].data
        assert "Component" not in data
        assert "Tool" not in data
        assert "ExecutionStage" not in data


# ---------------------------------------------------------------------------
# Payload: stage-aware Inputs / Outputs (input vs output escalation)
# ---------------------------------------------------------------------------


class TestStageAwarePayload:
    def test_pre_maps_content_to_inputs_only(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mi:
            with _published_context(
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.PRE,
                component="my_tool",
            ):
                _action().handle_validation_result(_failed(), {"a": 1}, "g")
        data = mi.call_args[0][0].data
        assert json.loads(data["Inputs"]) == {"a": 1}
        assert data["Outputs"] == ""  # no output at PRE
        assert data["ExecutionStage"] == "PreExecution"
        assert data["ToolInputs"] == data["Inputs"]
        assert data["ToolOutputs"] == data["Outputs"]

    def test_post_maps_output_to_outputs_and_input_to_inputs(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mi:
            with _published_context(
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.POST,
                component="my_tool",
                input_payload=json.dumps({"in": 1}),
            ):
                _action().handle_validation_result(_failed(), {"out": 2}, "g")
        data = mi.call_args[0][0].data
        assert json.loads(data["Outputs"]) == {"out": 2}
        assert json.loads(data["Inputs"]) == {"in": 1}
        assert data["ExecutionStage"] == "PostExecution"
        assert data["Tool"] == "my_tool"
        assert data["Component"] == "my_tool"
        assert data["ToolInputs"] == data["Inputs"]
        assert data["ToolOutputs"] == data["Outputs"]

    def test_post_without_input_payload_leaves_inputs_empty(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mi:
            with _published_context(
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.POST,
                component="my_tool",
            ):
                _action().handle_validation_result(_failed(), {"out": 2}, "g")
        data = mi.call_args[0][0].data
        assert json.loads(data["Outputs"]) == {"out": 2}
        assert data["Inputs"] == ""
        assert data["ToolInputs"] == data["Inputs"]
        assert data["ToolOutputs"] == data["Outputs"]


# ---------------------------------------------------------------------------
# Payload: GuardrailDescription and GuardrailResult
# ---------------------------------------------------------------------------


class TestGuardrailDescription:
    def test_description_comes_from_context(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mi:
            with _published_context(description="Detects PII emails"):
                _action().handle_validation_result(_failed("PII detected"), "x", "g")
        data = mi.call_args[0][0].data
        assert data["GuardrailDescription"] == "Detects PII emails"
        assert data["GuardrailResult"] == "PII detected"

    def test_description_empty_when_absent(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mi:
            _action().handle_validation_result(_failed("PII detected"), "x", "g")
        data = mi.call_args[0][0].data
        assert data["GuardrailDescription"] == ""
        assert data["GuardrailResult"] == "PII detected"


# ---------------------------------------------------------------------------
# Payload: TenantName / AgentTrace from UiPathConfig
# ---------------------------------------------------------------------------


class TestConfigDerivedFields:
    def test_tenant_name_from_config(self) -> None:
        cfg = SimpleNamespace(tenant_name="MyTenant", base_url=None)
        with _patched(resume={"action": "Approve", "data": {}}, config=cfg) as mi:
            _action().handle_validation_result(_failed(), "x", "g")
        assert mi.call_args[0][0].data["TenantName"] == "MyTenant"

    def test_tenant_name_falls_back_to_base_url(self) -> None:
        cfg = SimpleNamespace(
            tenant_name=None, base_url="https://alpha.uipath.com/Org/MyTenant"
        )
        with _patched(resume={"action": "Approve", "data": {}}, config=cfg) as mi:
            _action().handle_validation_result(_failed(), "x", "g")
        assert mi.call_args[0][0].data["TenantName"] == "MyTenant"

    def test_tenant_name_omitted_when_unresolvable(self) -> None:
        cfg = SimpleNamespace(tenant_name=None, base_url=None)
        with _patched(resume={"action": "Approve", "data": {}}, config=cfg) as mi:
            _action().handle_validation_result(_failed(), "x", "g")
        assert "TenantName" not in mi.call_args[0][0].data

    def test_agent_trace_deployed_url(self) -> None:
        cfg = SimpleNamespace(
            tenant_name="T",
            base_url="https://alpha.uipath.com/Org/T",
            organization_id="org-123",
            is_studio_project=False,
            folder_key="folder-1",
            process_uuid="proc-1",
            trace_id="trace-1",
            project_key="project-1",
            process_version="1.0.0",
        )
        with _patched(resume={"action": "Approve", "data": {}}, config=cfg) as mi:
            _action().handle_validation_result(_failed(), "x", "g")
        trace = mi.call_args[0][0].data["AgentTrace"]
        assert trace.startswith("https://alpha.uipath.com/org-123/agents_/deployed/")
        assert "/traces/trace-1" in trace

    def test_agent_trace_omitted_when_ids_missing(self) -> None:
        cfg = SimpleNamespace(
            tenant_name="T",
            base_url="https://alpha.uipath.com/Org/T",
            organization_id="org-123",
        )
        with _patched(resume={"action": "Approve", "data": {}}, config=cfg) as mi:
            _action().handle_validation_result(_failed(), "x", "g")
        assert "AgentTrace" not in mi.call_args[0][0].data


# ---------------------------------------------------------------------------
# Response handling: Approve / Reject / modify
# ---------------------------------------------------------------------------


class TestResponseHandling:
    def test_approve_with_reviewed_inputs_string(self) -> None:
        resume = {"action": "Approve", "data": {"ReviewedInputs": "clean topic"}}
        with _patched(resume=resume):
            result = _action().handle_validation_result(_failed(), "dirty topic", "g")
        assert result == "clean topic"

    def test_approve_with_reviewed_inputs_dict_payload(self) -> None:
        reviewed = {"email": "[redacted]"}
        resume = {"action": "Approve", "data": {"ReviewedInputs": json.dumps(reviewed)}}
        with _patched(resume=resume):
            result = _action().handle_validation_result(
                _failed(), {"email": "a@b.com"}, "g"
            )
        assert result == reviewed

    def test_approve_without_reviewed_inputs_returns_none(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}):
            result = _action().handle_validation_result(_failed(), "topic", "g")
        assert result is None

    def test_reject_raises_block_exception_with_reason(self) -> None:
        resume = {"action": "Reject", "data": {"Reason": "contains PII"}}
        with _patched(resume=resume):
            with pytest.raises(GuardrailBlockException) as exc_info:
                _action().handle_validation_result(_failed(), "topic", "PII guard")
        assert "contains PII" in exc_info.value.detail
        assert "PII guard" in exc_info.value.title

    def test_reject_falls_back_to_result_reason(self) -> None:
        resume = {"action": "Reject", "data": {}}
        with _patched(resume=resume):
            with pytest.raises(GuardrailBlockException) as exc_info:
                _action().handle_validation_result(_failed("the reason"), "topic", "g")
        assert "the reason" in exc_info.value.detail

    def test_post_approve_reads_reviewed_outputs(self) -> None:
        resume = {"action": "Approve", "data": {"ReviewedOutputs": "clean output"}}
        with _patched(resume=resume):
            with _published_context(
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.POST,
                component="t",
            ):
                result = _action().handle_validation_result(
                    _failed(), "dirty output", "g"
                )
        assert result == "clean output"

    def test_post_approve_ignores_reviewed_inputs(self) -> None:
        resume = {"action": "Approve", "data": {"ReviewedInputs": "ignored"}}
        with _patched(resume=resume):
            with _published_context(stage=GuardrailExecutionStage.POST):
                result = _action().handle_validation_result(_failed(), "orig", "g")
        assert result is None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNormalizeEscalationResult:
    def test_none_defaults_to_approve(self) -> None:
        assert _normalize_escalation_result(None) == ("Approve", {})

    def test_dict_with_action_key(self) -> None:
        outcome, data = _normalize_escalation_result(
            {"action": "Reject", "data": {"Reason": "x"}}
        )
        assert outcome == "Reject"
        assert data == {"Reason": "x"}

    def test_dict_with_capitalized_action_key(self) -> None:
        outcome, _ = _normalize_escalation_result({"Action": "Approve"})
        assert outcome == "Approve"

    def test_dict_without_nested_data_uses_self(self) -> None:
        outcome, data = _normalize_escalation_result(
            {"action": "Approve", "ReviewedInputs": "v"}
        )
        assert outcome == "Approve"
        assert data == {"action": "Approve", "ReviewedInputs": "v"}

    def test_object_with_action_and_data(self) -> None:
        raw = SimpleNamespace(action="Reject", data={"Reason": "no"})
        assert _normalize_escalation_result(raw) == ("Reject", {"Reason": "no"})

    def test_object_without_action_defaults_to_approve(self) -> None:
        raw = SimpleNamespace(action=None, data=None)
        assert _normalize_escalation_result(raw) == ("Approve", {})


class TestCoerceReviewed:
    def test_want_dict_parses_json_object(self) -> None:
        assert _coerce_reviewed(json.dumps({"a": 1}), want_dict=True) == {"a": 1}

    def test_want_dict_non_json_returns_as_is(self) -> None:
        assert _coerce_reviewed("not json", want_dict=True) == "not json"

    def test_want_str_returns_as_is(self) -> None:
        assert _coerce_reviewed("clean", want_dict=False) == "clean"


class TestExecutionStageLabel:
    def test_pre(self) -> None:
        assert _execution_stage_label(GuardrailExecutionStage.PRE) == "PreExecution"

    def test_post(self) -> None:
        assert _execution_stage_label(GuardrailExecutionStage.POST) == "PostExecution"

    def test_pre_and_post_returns_none(self) -> None:
        assert _execution_stage_label(GuardrailExecutionStage.PRE_AND_POST) is None

    def test_none(self) -> None:
        assert _execution_stage_label(None) is None


class TestResolveTenantName:
    def test_prefers_config_tenant_name(self) -> None:
        cfg = SimpleNamespace(tenant_name="FromEnv", base_url="https://x/o/FromUrl")
        with patch.object(escalate_module, "UiPathConfig", cfg):
            assert _resolve_tenant_name() == "FromEnv"

    def test_falls_back_to_url(self) -> None:
        cfg = SimpleNamespace(tenant_name=None, base_url="https://x/org/tenant")
        with patch.object(escalate_module, "UiPathConfig", cfg):
            assert _resolve_tenant_name() == "tenant"

    def test_returns_none_when_unresolvable(self) -> None:
        cfg = SimpleNamespace(tenant_name=None, base_url=None)
        with patch.object(escalate_module, "UiPathConfig", cfg):
            assert _resolve_tenant_name() is None


# ---------------------------------------------------------------------------
# Escalation target: recipient (TaskRecipient) — the HITL-native typed target
# ---------------------------------------------------------------------------


class TestRecipient:
    @pytest.mark.parametrize(
        "recipient_type,value",
        [
            (TaskRecipientType.USER_ID, "user-guid-1"),
            (TaskRecipientType.GROUP_ID, "group-guid-1"),
            (TaskRecipientType.EMAIL, "reviewer@x.com"),
            (TaskRecipientType.GROUP_NAME, "Reviewers"),
        ],
    )
    def test_recipient_passed_through(
        self, recipient_type: TaskRecipientType, value: str
    ) -> None:
        recipient = TaskRecipient(type=recipient_type, value=value)
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            _action(recipient=recipient).handle_validation_result(_failed(), "x", "g")
        payload = mock_interrupt.call_args[0][0]
        assert payload.recipient == recipient
        assert payload.recipient.type == recipient_type
        assert payload.recipient.value == value

    def test_recipient_defaults_to_none(self) -> None:
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            _action(assignee="user@x.com").handle_validation_result(_failed(), "x", "g")
        payload = mock_interrupt.call_args[0][0]
        assert payload.recipient is None
        assert payload.assignee == "user@x.com"

    def test_assignee_and_recipient_coexist(self) -> None:
        recipient = TaskRecipient(type=TaskRecipientType.GROUP_NAME, value="Reviewers")
        with _patched(resume={"action": "Approve", "data": {}}) as mock_interrupt:
            _action(
                assignee="user@x.com", recipient=recipient
            ).handle_validation_result(_failed(), "x", "g")
        payload = mock_interrupt.call_args[0][0]
        assert payload.assignee == "user@x.com"
        assert payload.recipient == recipient
