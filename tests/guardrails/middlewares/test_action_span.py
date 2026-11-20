"""The middleware records the action it takes on a violation as a guardrailAction span."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.prebuilt.tool_node import ToolCallRequest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails.decorators import (
    BlockAction,
    LogAction,
    LoggingSeverityLevel,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.guardrails import GuardrailAction, GuardrailExecutionStage
from uipath_langchain.guardrails.enums import PIIDetectionEntityType
from uipath_langchain.guardrails.middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    _grouping,
)
from uipath_langchain.guardrails.middlewares._spans import action_name
from uipath_langchain.guardrails.models import PIIDetectionEntity

_SERVICE_SPAN_ID = "00000000-0000-0000-1234-567890abcdef"
_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED,
    reason="PII detected",
    span_id=_SERVICE_SPAN_ID,
)
_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)


@pytest.fixture
def recorded_spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    monkeypatch.setattr(_grouping, "_state", _grouping._GroupingState())
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()


def _check(action: GuardrailAction, result: GuardrailValidationResult) -> None:
    """Run an LLM-input check under a span named ``caller``."""
    middleware = UiPathPIIDetectionMiddleware(
        scopes=[GuardrailScope.LLM],
        action=action,
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
    )
    with (
        patch.object(middleware, "_evaluate_guardrail", return_value=result),
        trace.get_tracer("test").start_as_current_span("caller"),
    ):
        middleware._check_messages([HumanMessage(content="mail me at a@b.com")])


def _action_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if (span.attributes or {}).get("span_type") == "guardrailAction"
    ]


class _Recording(GuardrailAction):
    def __init__(self, raise_exc: BaseException | None = None) -> None:
        self._raise = raise_exc

    def handle_validation_result(
        self, result: Any, data: Any, guardrail_name: str
    ) -> Any:
        if self._raise is not None:
            raise self._raise
        return None


class TestGuardrailActionSpan:
    def test_block_records_an_error_span_linked_to_the_service_span(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        with pytest.raises(AgentRuntimeError):
            _check(BlockAction(), _FAILED)

        [span] = _action_spans(recorded_spans)
        assert span.name == "Block"
        attributes = dict(span.attributes or {})
        assert attributes["type"] == "guardrailAction"
        assert attributes["uipath.custom_instrumentation"] is True
        assert attributes["guardrailName"] == "PII Detection"
        assert attributes["action"] == "Block"
        assert attributes["reason"] == "PII detected"
        assert attributes["guardrailSpanId"] == _SERVICE_SPAN_ID
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "Guardrail violation: PII detected"
        caller = next(
            s for s in recorded_spans.get_finished_spans() if s.name == "caller"
        )
        assert span.parent is not None
        assert span.parent.span_id == caller.context.span_id

    def test_log_records_an_ok_span_with_its_severity(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        _check(LogAction(severity_level=LoggingSeverityLevel.ERROR), _FAILED)

        [span] = _action_spans(recorded_spans)
        assert span.name == "Log"
        attributes = dict(span.attributes or {})
        assert attributes["action"] == "Log"
        assert attributes["severityLevel"] == "Error"
        assert attributes["reason"] == "PII detected"
        assert span.status.status_code == StatusCode.OK

    def test_passed_result_records_no_action_span(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        _check(BlockAction(), _PASSED)

        assert _action_spans(recorded_spans) == []

    def test_suspending_escalation_ends_the_span_without_error(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        escalation = _Recording(raise_exc=GraphInterrupt())
        with pytest.raises(GraphInterrupt):
            _check(escalation, _FAILED)

        [span] = _action_spans(recorded_spans)
        assert span.status.status_code == StatusCode.OK

    def test_missing_service_span_id_is_omitted(
        self, recorded_spans: InMemorySpanExporter
    ) -> None:
        without_span_id = _FAILED.model_copy(update={"span_id": None})
        _check(LogAction(), without_span_id)

        [span] = _action_spans(recorded_spans)
        assert "guardrailSpanId" not in (span.attributes or {})


async def test_deterministic_rules_record_their_action_too(
    recorded_spans: InMemorySpanExporter,
) -> None:
    rule = UiPathDeterministicGuardrailMiddleware(
        tools=["get_weather"],
        rules=[lambda args: args.get("city") == "London"],
        action=LogAction(),
        stage=GuardrailExecutionStage.PRE,
        name="London Rule",
    )
    [wrapper] = list(rule)
    request = ToolCallRequest(
        tool_call={"id": "tc1", "name": "get_weather", "args": {"city": "London"}},
        tool=MagicMock(),
        state={},
        runtime=MagicMock(),
    )

    async def tool(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="sunny", tool_call_id="tc1")

    await wrapper.awrap_tool_call(request, tool)

    [span] = _action_spans(recorded_spans)
    attributes = dict(span.attributes or {})
    assert attributes["guardrailName"] == "London Rule"
    assert attributes["action"] == "Log"
    assert attributes["reason"] == "Rule 1 detected violation"


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (BlockAction(), "Block"),
        (LogAction(), "Log"),
        (_Recording(), "_Recording"),
    ],
)
def test_action_name(action: Any, expected: str) -> None:
    assert action_name(action) == expected
