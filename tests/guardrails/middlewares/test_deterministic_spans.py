"""Deterministic rules record their check as a guardrailEvaluation span."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from uipath.platform.guardrails.decorators import BlockAction, LogAction

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.guardrails import GuardrailAction, GuardrailExecutionStage
from uipath_langchain.guardrails.middlewares import (
    UiPathDeterministicGuardrailMiddleware,
)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()


async def _call_weather(action: GuardrailAction, city: str) -> None:
    rule = UiPathDeterministicGuardrailMiddleware(
        tools=["get_weather"],
        rules=[lambda args: args.get("city") == "London"],
        action=action,
        stage=GuardrailExecutionStage.PRE,
        name="London Rule",
        description="Flags London",
    )
    [wrapper] = list(rule)
    request = ToolCallRequest(
        tool_call={"id": "tc1", "name": "get_weather", "args": {"city": city}},
        tool=MagicMock(),
        state={},
        runtime=MagicMock(),
    )

    async def tool(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="sunny", tool_call_id="tc1")

    await wrapper.awrap_tool_call(request, tool)


def _named(spans: InMemorySpanExporter, name: str) -> ReadableSpan:
    [span] = [s for s in spans.get_finished_spans() if s.name == name]
    return span


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


async def test_a_failed_rule_records_its_evaluation_and_links_the_action(
    spans: InMemorySpanExporter,
) -> None:
    await _call_weather(LogAction(), "London")

    evaluation = _attributes(_named(spans, "London Rule"))
    assert evaluation["span_type"] == "guardrailEvaluation"
    assert evaluation["guardrailType"] == "custom"
    assert evaluation["guardrailDescription"] == "Flags London"
    assert evaluation["validationResult"] == "VALIDATION_FAILED"
    assert evaluation["details"] == ("Rule 1 detected violation",)
    span_id = format(_named(spans, "London Rule").context.span_id, "016x")
    log = _attributes(_named(spans, "Log"))
    assert log["guardrailSpanId"] == f"00000000-0000-0000-{span_id[:4]}-{span_id[4:]}"


async def test_a_passing_rule_records_its_evaluation_only(
    spans: InMemorySpanExporter,
) -> None:
    await _call_weather(LogAction(), "Paris")

    evaluation = _attributes(_named(spans, "London Rule"))
    assert evaluation["validationResult"] == "PASSED"
    assert [s.name for s in spans.get_finished_spans()] == ["London Rule"]


async def test_a_blocking_rule_marks_its_action_as_failed(
    spans: InMemorySpanExporter,
) -> None:
    with pytest.raises(AgentRuntimeError):
        await _call_weather(BlockAction(), "London")

    assert _named(spans, "Block").status.status_code == StatusCode.ERROR
    assert _named(spans, "London Rule").status.status_code == StatusCode.OK
