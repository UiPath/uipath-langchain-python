"""Guardrail checks carry the uipath_governance.* contract on their spans."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationErrorType,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails.decorators import BlockAction, LogAction

from uipath_langchain.guardrails import (
    EscalateAction,
    GuardrailAction,
    GuardrailExecutionStage,
)
from uipath_langchain.guardrails.enums import PIIDetectionEntityType
from uipath_langchain.guardrails.middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    _grouping,
)
from uipath_langchain.guardrails.middlewares._enforcement import (
    EnforcementContext,
    container_attributes,
)
from uipath_langchain.guardrails.models import PIIDetectionEntity

_PRE = GuardrailExecutionStage.PRE
_POST = GuardrailExecutionStage.POST

_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="PII detected"
)
_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)


def _contract(span: ReadableSpan) -> dict[str, Any]:
    return {
        key.removeprefix("uipath_governance."): value
        for key, value in (span.attributes or {}).items()
        if key.startswith("uipath_governance.")
    }


def _of_type(exporter: InMemorySpanExporter, span_type: str) -> list[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if (span.attributes or {}).get("span_type") == span_type
    ]


@pytest.fixture
def recorded_spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    monkeypatch.setattr(_grouping, "_state", _grouping._GroupingState())
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()


class _CustomAction(GuardrailAction):
    def handle_validation_result(
        self, result: Any, data: Any, guardrail_name: str
    ) -> Any:
        return None


@pytest.mark.parametrize(
    ("scope", "stage", "hook"),
    [
        (GuardrailScope.AGENT, _PRE, "BEFORE_AGENT"),
        (GuardrailScope.AGENT, _POST, "AFTER_AGENT"),
        (GuardrailScope.LLM, _PRE, "BEFORE_MODEL"),
        (GuardrailScope.LLM, _POST, "AFTER_MODEL"),
        (GuardrailScope.TOOL, _PRE, "TOOL_CALL"),
        (GuardrailScope.TOOL, _POST, "AFTER_TOOL"),
    ],
)
def test_each_stage_container_carries_its_version_and_hook(
    scope: GuardrailScope, stage: GuardrailExecutionStage, hook: str
) -> None:
    expected = {"uipath_governance.version": "1", "uipath_governance.hook": hook}
    assert container_attributes(scope, stage) == expected
    assert _grouping._STAGES[(scope, stage)].attributes == expected


class FilterAction:
    """Stands in for a filter action: only its type name is read."""


class TestEnforcementContext:
    @staticmethod
    def _context(action: Any, **overrides: Any) -> EnforcementContext:
        fields: dict[str, Any] = {
            "scope": GuardrailScope.LLM,
            "stage": _PRE,
            "action": action,
            "guardrail_name": "PII Detection",
            "guardrail_type": "pii_detection",
        }
        return EnforcementContext(**(fields | overrides))

    @pytest.mark.parametrize(
        ("action", "action_applied"),
        [
            (BlockAction(), "DENY"),
            (LogAction(), "AUDIT"),
            (EscalateAction(app_name="app", app_folder_path="f"), "HITL"),
            (FilterAction(), "FILTER"),
        ],
    )
    def test_a_violation_applies_the_action(
        self, action: Any, action_applied: str
    ) -> None:
        assert self._context(action).attributes(_FAILED) == {
            "uipath_governance.policy_id": "",
            "uipath_governance.policy_name": "",
            "uipath_governance.hook": "BEFORE_MODEL",
            "uipath_governance.evaluator_result": "MATCHED",
            "uipath_governance.action_applied": action_applied,
            "uipath_governance.mode": "ENFORCE",
            "uipath_governance.version": "1",
            "uipath_governance.evidence": "PII detected",
            "uipath_governance.guardrail_type": "pii_detection",
            "uipath_governance.guardrail_name": "PII Detection",
        }

    def test_a_pass_allows_with_default_evidence(self) -> None:
        attributes = self._context(LogAction()).attributes(_PASSED)

        assert attributes["uipath_governance.evaluator_result"] == "NOT_MATCHED"
        assert attributes["uipath_governance.action_applied"] == "ALLOW"
        assert attributes["uipath_governance.mode"] == "ENFORCE"
        assert attributes["uipath_governance.evidence"] == "PII Detection passed"

    def test_a_check_that_failed_closed_is_an_error(self) -> None:
        result = _FAILED.model_copy(
            update={"error_type": GuardrailValidationErrorType.PROVIDER_ERROR}
        )

        attributes = self._context(BlockAction()).attributes(result)

        assert attributes["uipath_governance.evaluator_result"] == "ERROR"
        assert attributes["uipath_governance.action_applied"] == "DENY"

    def test_an_unknown_action_writes_nothing(self) -> None:
        context = self._context(_CustomAction())

        assert context.attributes(_FAILED) == {}
        assert context.evaluation_options() == {}

    def test_an_unknown_stage_writes_nothing(self) -> None:
        context = self._context(BlockAction(), scope=None, stage=None)

        assert context.attributes(_FAILED) == {}
        assert context.evaluation_options() == {}

    def test_evaluation_options_use_the_service_values(self) -> None:
        context = self._context(BlockAction(), scope=GuardrailScope.TOOL, stage=_POST)

        assert context.evaluation_options() == {
            "scope": GuardrailScope.TOOL,
            "execution_stage": "postExecution",
            "action_type": "Block",
        }


def _pii(scopes: list[GuardrailScope], action: Any) -> UiPathPIIDetectionMiddleware:
    return UiPathPIIDetectionMiddleware(
        scopes=scopes,
        action=action,
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
        tools=["get_weather"] if GuardrailScope.TOOL in scopes else None,
    )


def _request(args: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"id": "tc1", "name": "get_weather", "args": args},
        tool=MagicMock(),
        state={},
        runtime=MagicMock(),
    )


async def _sunny(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="sunny", tool_call_id="tc1")


class TestBuiltInChecks:
    def test_the_service_is_told_the_scope_stage_and_action(self) -> None:
        middleware = _pii([GuardrailScope.LLM], LogAction())
        uipath = MagicMock()
        uipath.guardrails.evaluate_guardrail.return_value = _PASSED

        with patch.object(middleware, "_get_uipath", return_value=uipath):
            middleware._check_messages(
                [HumanMessage(content="a@b.com")],
                scope=GuardrailScope.LLM,
                stage=_POST,
            )

        call = uipath.guardrails.evaluate_guardrail.call_args
        assert call.kwargs["scope"] == GuardrailScope.LLM
        assert call.kwargs["execution_stage"] == "postExecution"
        assert call.kwargs["action_type"] == "Log"

    async def test_a_tool_check_is_told_its_stage(self) -> None:
        middleware = _pii([GuardrailScope.TOOL], BlockAction())
        uipath = MagicMock()
        uipath.guardrails.evaluate_guardrail.return_value = _PASSED

        with patch.object(middleware, "_get_uipath", return_value=uipath):
            await middleware._run_tool_guardrail(_request({"city": "a"}), _sunny)

        stages = [
            call.kwargs["execution_stage"]
            for call in uipath.guardrails.evaluate_guardrail.call_args_list
        ]
        assert stages == ["preExecution", "postExecution"]

    def test_an_unknown_action_tells_the_service_nothing(self) -> None:
        middleware = _pii([GuardrailScope.LLM], _CustomAction())
        uipath = MagicMock()
        uipath.guardrails.evaluate_guardrail.return_value = _PASSED

        with patch.object(middleware, "_get_uipath", return_value=uipath):
            middleware._check_messages(
                [HumanMessage(content="a@b.com")],
                scope=GuardrailScope.LLM,
                stage=_PRE,
            )

        assert uipath.guardrails.evaluate_guardrail.call_args.kwargs == {}

    def test_the_action_span_carries_the_decision(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        middleware = _pii([GuardrailScope.LLM], LogAction())

        with patch.object(middleware, "_evaluate_guardrail", return_value=_FAILED):
            middleware._check_messages(
                [HumanMessage(content="a@b.com")],
                scope=GuardrailScope.LLM,
                stage=_PRE,
            )

        [span] = _of_type(recorded_spans, "guardrailAction")
        decision = _contract(span)
        assert decision["hook"] == "BEFORE_MODEL"
        assert decision["evaluator_result"] == "MATCHED"
        assert decision["action_applied"] == "AUDIT"
        assert decision["guardrail_type"] == "pii_detection"


class TestDeterministicChecks:
    @staticmethod
    def _rule(stage: GuardrailExecutionStage) -> Any:
        [wrapper] = list(
            UiPathDeterministicGuardrailMiddleware(
                tools=["get_weather"],
                rules=[lambda args: args.get("city") == "London"],
                action=LogAction(),
                stage=stage,
                name="London Rule",
            )
        )
        return wrapper

    async def test_a_violation_stamps_evaluation_and_action(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        await self._rule(_PRE).awrap_tool_call(_request({"city": "London"}), _sunny)

        [evaluation] = _of_type(recorded_spans, "guardrailEvaluation")
        [action] = _of_type(recorded_spans, "guardrailAction")
        expected = {
            "policy_id": "",
            "policy_name": "",
            "hook": "TOOL_CALL",
            "evaluator_result": "MATCHED",
            "action_applied": "AUDIT",
            "mode": "ENFORCE",
            "version": "1",
            "evidence": "Rule 1 detected violation",
            "guardrail_type": "custom",
            "guardrail_name": "London Rule",
        }
        assert _contract(evaluation) == expected
        assert _contract(action) == expected

    async def test_a_pass_stamps_the_evaluation(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        await self._rule(_PRE).awrap_tool_call(_request({"city": "Paris"}), _sunny)

        [evaluation] = _of_type(recorded_spans, "guardrailEvaluation")
        decision = _contract(evaluation)
        assert decision["evaluator_result"] == "NOT_MATCHED"
        assert decision["action_applied"] == "ALLOW"
        assert decision["evidence"] == "Rules passed: Rule 1"
        assert _of_type(recorded_spans, "guardrailAction") == []

    async def test_a_rule_that_raises_is_an_error(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        def broken(args: dict[str, Any]) -> bool:
            raise ValueError("boom")

        [wrapper] = list(
            UiPathDeterministicGuardrailMiddleware(
                tools=["get_weather"],
                rules=[lambda args: args.get("city") == "London", broken],
                action=LogAction(),
                stage=_PRE,
                name="London Rule",
            )
        )

        await wrapper.awrap_tool_call(_request({"city": "London"}), _sunny)

        [evaluation] = _of_type(recorded_spans, "guardrailEvaluation")
        [action] = _of_type(recorded_spans, "guardrailAction")
        for span in (evaluation, action):
            decision = _contract(span)
            assert decision["evaluator_result"] == "ERROR"
            assert decision["action_applied"] == "AUDIT"
            assert "Rule 2 raised exception: boom" in decision["evidence"]

    async def test_a_post_check_is_after_the_tool(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        await self._rule(_POST).awrap_tool_call(_request({"city": "London"}), _sunny)

        [evaluation] = _of_type(recorded_spans, "guardrailEvaluation")
        assert _contract(evaluation)["hook"] == "AFTER_TOOL"
