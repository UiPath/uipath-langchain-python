"""Separate guardrail middlewares share one container span per stage."""

import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_ancestor_spans,
    get_current_span,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    StatusCode,
    TraceFlags,
)
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.core.tracing.span_utils import UiPathSpanUtils
from uipath.platform.guardrails.decorators import BlockAction, LogAction

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.guardrails import GuardrailExecutionStage
from uipath_langchain.guardrails.enums import PIIDetectionEntityType
from uipath_langchain.guardrails.middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
    _base,
    _grouping,
)
from uipath_langchain.guardrails.middlewares._grouping import (
    enable_guardrail_grouping,
    guardrail_stage,
    is_exported,
)
from uipath_langchain.guardrails.models import PIIDetectionEntity

_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="found"
)
_STATE = {"messages": [HumanMessage(content="mail me at a@b.com")]}
_OPENINFERENCE = "openinference.instrumentation.langchain"


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    monkeypatch.setattr(_grouping, "_state", _grouping._GroupingState())
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    exporter.provider = provider  # type: ignore[attr-defined]
    yield exporter
    UiPathSpanUtils.register_current_span_provider(None)
    UiPathSpanUtils.register_current_span_ancestors_provider(None)
    provider.shutdown()


class _Graph:
    """Plays LangGraph: a graph span and node spans started in order."""

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self.exporter = exporter
        self._tracer = exporter.provider.get_tracer(_OPENINFERENCE)  # type: ignore[attr-defined]
        self.current: Span | None = None
        self.span = self._tracer.start_span("LangGraph")

    @contextmanager
    def node(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
        start_time: int | None = None,
    ) -> Iterator[Span]:
        span = self._tracer.start_span(
            name,
            context=trace.set_span_in_context(parent or self.span),
            attributes=attributes,
            start_time=start_time,
        )
        previous, self.current = self.current, span
        try:
            yield span
        finally:
            self.current = previous
            span.end()

    def finish(self) -> list[ReadableSpan]:
        self.span.end()
        return list(self.exporter.get_finished_spans())


@pytest.fixture
def graph(exporter: InMemorySpanExporter) -> _Graph:
    fake = _Graph(exporter)
    enable_guardrail_grouping(exporter.provider, lambda: fake.current)  # type: ignore[attr-defined]
    return fake


def _hook(middleware: Any, name: str) -> Any:
    [hook] = [h for h in middleware if h.name.endswith(name)]
    return hook


def _named(spans: list[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


def _children(spans: list[ReadableSpan], parent: ReadableSpan) -> list[str]:
    return [
        s.name
        for s in spans
        if s.parent is not None and s.parent.span_id == parent.context.span_id
    ]


def _pii(**kwargs: Any) -> UiPathPIIDetectionMiddleware:
    return UiPathPIIDetectionMiddleware(
        scopes=kwargs.pop("scopes", [GuardrailScope.LLM]),
        action=kwargs.pop("action", LogAction()),
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
        **kwargs,
    )


def _rule(stage: GuardrailExecutionStage) -> UiPathDeterministicGuardrailMiddleware:
    return UiPathDeterministicGuardrailMiddleware(
        tools=["get_weather"],
        rules=[lambda *args: True],
        action=LogAction(),
        stage=stage,
        name="Rule",
    )


def _tool_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"id": "tc1", "name": "get_weather", "args": {"city": "London"}},
        tool=MagicMock(),
        state={},
        runtime=MagicMock(),
    )


class TestStages:
    async def test_guardrail_nodes_of_one_stage_share_a_container(
        self, graph: _Graph
    ) -> None:
        pii = _pii()
        injection = UiPathPromptInjectionMiddleware(action=LogAction())
        pii_hook = _hook(pii, "_before_model")
        injection_hook = _hook(injection, "_before_model")

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            patch.object(injection, "_evaluate_guardrail", return_value=_FAILED),
        ):
            with graph.node(f"{pii_hook.name}.before_model"):
                await pii_hook.abefore_model(_STATE, MagicMock())
            with graph.node(f"{injection_hook.name}.before_model"):
                await injection_hook.abefore_model(_STATE, MagicMock())
        with graph.node("model") as model:
            model_start = model.start_time  # type: ignore[attr-defined]
        spans = graph.finish()

        [container] = _named(spans, "LLM input guardrail check")
        assert (container.attributes or {})["span_type"] == "llmPreGuardrails"
        assert container.parent is not None
        assert container.parent.span_id == graph.span.get_span_context().span_id
        assert _children(spans, container) == ["Log", "Log"]
        assert container.end_time == model_start
        assert container.status.status_code == StatusCode.OK

    async def test_a_container_never_ends_before_its_last_check(
        self, graph: _Graph
    ) -> None:
        pii = _pii()
        hook = _hook(pii, "_before_model")

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            graph.node(f"{hook.name}.before_model"),
        ):
            await hook.abefore_model(_STATE, MagicMock())
        # A coarse clock stamps the next node's start before the check ended.
        with graph.node("model", start_time=time.time_ns() - 10_000_000):
            pass
        spans = graph.finish()

        [container] = _named(spans, "LLM input guardrail check")
        [log] = _named(spans, "Log")
        assert container.end_time is not None
        assert log.end_time is not None
        assert container.end_time >= log.end_time

    async def test_the_container_carries_the_execution_id(self, graph: _Graph) -> None:
        pii = _pii()
        hook = _hook(pii, "_before_model")

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            graph.node(
                f"{hook.name}.before_model", attributes={"execution.id": "exec-1"}
            ),
        ):
            await hook.abefore_model(_STATE, MagicMock())
        spans = graph.finish()

        [container] = _named(spans, "LLM input guardrail check")
        assert (container.attributes or {})["execution.id"] == "exec-1"

    async def test_a_new_stage_gets_its_own_container(self, graph: _Graph) -> None:
        pii = _pii(scopes=[GuardrailScope.AGENT, GuardrailScope.LLM])
        agent_hook = _hook(pii, "_before_agent")
        model_hook = _hook(pii, "_before_model")

        with patch.object(pii, "_evaluate_guardrail", return_value=_FAILED):
            with graph.node(f"{agent_hook.name}.before_agent"):
                await agent_hook.abefore_agent(_STATE, MagicMock())
            with graph.node(f"{model_hook.name}.before_model"):
                await model_hook.abefore_model(_STATE, MagicMock())
        spans = graph.finish()

        assert len(_named(spans, "Agent input guardrail check")) == 1
        assert len(_named(spans, "LLM input guardrail check")) == 1

    async def test_another_node_between_guardrails_splits_the_stage(
        self, graph: _Graph
    ) -> None:
        pii = _pii()
        injection = UiPathPromptInjectionMiddleware(action=LogAction())
        pii_hook = _hook(pii, "_before_model")
        injection_hook = _hook(injection, "_before_model")

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            patch.object(injection, "_evaluate_guardrail", return_value=_FAILED),
        ):
            with graph.node(f"{pii_hook.name}.before_model"):
                await pii_hook.abefore_model(_STATE, MagicMock())
            with graph.node("SummarizationMiddleware.before_model"):
                pass
            with graph.node(f"{injection_hook.name}.before_model"):
                await injection_hook.abefore_model(_STATE, MagicMock())
        spans = graph.finish()

        assert len(_named(spans, "LLM input guardrail check")) == 2

    async def test_llm_output_checks_get_their_container(self, graph: _Graph) -> None:
        pii = _pii()
        hook = _hook(pii, "_after_model")
        state = {"messages": [HumanMessage("hi"), AIMessage("mail a@b.com")]}

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            graph.node(f"{hook.name}.after_model"),
        ):
            await hook.aafter_model(state, MagicMock())
        spans = graph.finish()

        [container] = _named(spans, "LLM output guardrail check")
        assert _children(spans, container) == ["Log"]

    async def test_a_block_marks_the_container_as_failed(self, graph: _Graph) -> None:
        pii = _pii(action=BlockAction())
        hook = _hook(pii, "_before_model")

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            pytest.raises(AgentRuntimeError),
            graph.node(f"{hook.name}.before_model"),
        ):
            await hook.abefore_model(_STATE, MagicMock())
        spans = graph.finish()

        [container] = _named(spans, "LLM input guardrail check")
        assert container.status.status_code == StatusCode.ERROR

    async def test_a_grouping_failure_still_runs_the_check(self, graph: _Graph) -> None:
        pii = _pii(action=BlockAction())
        hook = _hook(pii, "_before_model")

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            patch.object(_grouping, "_shared_container", side_effect=RuntimeError),
            pytest.raises(AgentRuntimeError),
            graph.node(f"{hook.name}.before_model"),
        ):
            await hook.abefore_model(_STATE, MagicMock())
        spans = graph.finish()

        assert _named(spans, "LLM input guardrail check") == []
        assert len(_named(spans, "Block")) == 1

    async def test_a_stage_with_nothing_to_check_leaves_no_container(
        self, graph: _Graph
    ) -> None:
        pii = _pii()
        hook = _hook(pii, "_before_model")

        with (
            patch.object(pii, "_evaluate_guardrail") as evaluate,
            graph.node(f"{hook.name}.before_model"),
        ):
            await hook.abefore_model({"messages": []}, MagicMock())
        spans = graph.finish()

        evaluate.assert_not_called()
        assert not _named(spans, "LLM input guardrail check")


class TestTools:
    async def _call_tool(
        self, graph: _Graph, stage: GuardrailExecutionStage
    ) -> list[ReadableSpan]:
        [wrapper] = list(_rule(stage))
        with graph.node("tools") as tools:

            async def run_tool(request: ToolCallRequest) -> ToolMessage:
                with graph.node("get_weather", parent=tools):
                    return ToolMessage(content="sunny", tool_call_id="tc1")

            await wrapper.awrap_tool_call(_tool_request(), run_tool)
        return graph.finish()

    async def test_input_checks_close_when_the_tool_starts(self, graph: _Graph) -> None:
        spans = await self._call_tool(graph, GuardrailExecutionStage.PRE)

        [container] = _named(spans, "Tool input guardrail check")
        [tools] = _named(spans, "tools")
        [tool_span] = _named(spans, "get_weather")
        assert container.parent is not None
        assert container.parent.span_id == tools.context.span_id
        assert _children(spans, container) == ["Rule", "Log"]
        assert container.end_time == tool_span.start_time

    async def test_output_checks_close_when_the_tools_node_ends(
        self, graph: _Graph
    ) -> None:
        spans = await self._call_tool(graph, GuardrailExecutionStage.POST)

        [container] = _named(spans, "Tool output guardrail check")
        [tools] = _named(spans, "tools")
        [tool_span] = _named(spans, "get_weather")
        assert container.start_time is not None
        assert tool_span.end_time is not None
        assert container.start_time >= tool_span.end_time
        assert container.end_time == tools.end_time

    def test_a_container_closed_during_a_check_stays_closed(
        self, graph: _Graph
    ) -> None:
        _pii()
        with graph.node("tools") as tools:
            with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
                with graph.node("get_weather", parent=tools):  # closes it
                    pass
        spans = graph.finish()

        [container] = _named(spans, "Tool input guardrail check")
        [tool_span] = _named(spans, "get_weather")
        assert container.end_time == tool_span.start_time
        assert _grouping._state.containers == {}

    async def test_input_checks_close_when_the_output_stage_starts(
        self, graph: _Graph
    ) -> None:
        [pre] = list(_rule(GuardrailExecutionStage.PRE))
        [post] = list(_rule(GuardrailExecutionStage.POST))
        with graph.node("tools"):

            async def answer_without_the_tool(request: ToolCallRequest) -> ToolMessage:
                return ToolMessage(content="cached", tool_call_id="tc1")

            async def inner(request: ToolCallRequest) -> ToolMessage:
                return await post.awrap_tool_call(request, answer_without_the_tool)

            await pre.awrap_tool_call(_tool_request(), inner)
        spans = graph.finish()

        [pre_container] = _named(spans, "Tool input guardrail check")
        [post_container] = _named(spans, "Tool output guardrail check")
        assert pre_container.end_time is not None
        assert post_container.start_time is not None
        assert pre_container.end_time <= post_container.start_time

    async def test_a_failed_new_stage_keeps_the_previous_container_closable(
        self, graph: _Graph
    ) -> None:
        _pii()  # a guardrail in use turns grouping on
        with graph.node("tools") as tools:
            graph.current = tools
            with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
                pass
            with (
                patch.object(_grouping, "new_span", side_effect=RuntimeError),
                guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.POST),
            ):
                pass
        spans = graph.finish()

        [container] = _named(spans, "Tool input guardrail check")
        assert container.end_time == _named(spans, "tools")[0].end_time
        assert _grouping._state.containers == {}


class TestEnabling:
    async def test_nothing_is_grouped_without_a_closer(
        self, exporter: InMemorySpanExporter
    ) -> None:
        node = exporter.provider.get_tracer(_OPENINFERENCE).start_span("node")  # type: ignore[attr-defined]
        enable_guardrail_grouping(None, lambda: node)

        with guardrail_stage(GuardrailScope.LLM, GuardrailExecutionStage.PRE):
            pass
        node.end()

        assert _grouping._state.containers == {}
        assert not _grouping._state.enabled

    async def test_nothing_is_grouped_under_an_invalid_node_span(
        self, exporter: InMemorySpanExporter
    ) -> None:
        enable_guardrail_grouping(exporter.provider, lambda: trace.INVALID_SPAN)  # type: ignore[attr-defined]
        _pii()  # a guardrail in use turns grouping on

        with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
            pass

        assert _grouping._state.enabled
        assert _grouping._state.containers == {}

    def test_the_closer_is_added_once_per_trace_manager(
        self, exporter: InMemorySpanExporter
    ) -> None:
        manager = MagicMock()
        enable_guardrail_grouping(manager, lambda: None)
        _pii()
        enable_guardrail_grouping(manager, lambda: None)
        _pii()

        manager.add_span_processor.assert_called_once()

    def test_the_closer_comes_back_after_its_trace_manager_shuts_down(
        self, exporter: InMemorySpanExporter
    ) -> None:
        manager = MagicMock()
        enable_guardrail_grouping(manager, lambda: None)
        _pii()
        [closer] = [call.args[0] for call in manager.add_span_processor.call_args_list]

        closer.shutdown()  # what UiPathTraceManager.shutdown() does
        assert not _grouping._state.enabled
        enable_guardrail_grouping(manager, lambda: None)

        assert manager.add_span_processor.call_count == 2
        assert _grouping._state.enabled

    def test_a_shut_down_trace_manager_is_not_reused(
        self, exporter: InMemorySpanExporter
    ) -> None:
        manager = MagicMock()
        enable_guardrail_grouping(manager, lambda: None)
        _pii()
        [closer] = [call.args[0] for call in manager.add_span_processor.call_args_list]

        closer.shutdown()
        _pii()  # a middleware built before the next runtime factory

        manager.add_span_processor.assert_called_once()
        assert not _grouping._state.enabled

    async def test_nothing_is_grouped_under_a_sampled_out_node(
        self, exporter: InMemorySpanExporter
    ) -> None:
        node = NonRecordingSpan(
            SpanContext(
                trace_id=1, span_id=2, is_remote=False, trace_flags=TraceFlags(0)
            )
        )
        enable_guardrail_grouping(exporter.provider, lambda: node)  # type: ignore[attr-defined]
        _pii()

        with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
            pass

        assert _grouping._state.enabled
        assert _grouping._state.containers == {}

    def test_nothing_is_grouped_under_a_sampled_out_parent(
        self, exporter: InMemorySpanExporter
    ) -> None:
        # A sampler that keeps the node although its parent was dropped.
        provider = TracerProvider(sampler=ALWAYS_ON)
        parent = NonRecordingSpan(
            SpanContext(
                trace_id=1, span_id=2, is_remote=False, trace_flags=TraceFlags(0)
            )
        )
        node = provider.get_tracer(_OPENINFERENCE).start_span(
            "pii_detection_before_model", context=trace.set_span_in_context(parent)
        )
        enable_guardrail_grouping(exporter.provider, lambda: node)  # type: ignore[attr-defined]
        _pii()

        with guardrail_stage(GuardrailScope.LLM, GuardrailExecutionStage.PRE):
            pass

        assert node.is_recording()
        assert _grouping._state.containers == {}
        node.end()
        provider.shutdown()

    def test_the_closer_skips_the_lock_with_nothing_open(
        self, graph: _Graph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock = MagicMock()
        monkeypatch.setattr(_grouping._state, "lock", lock)
        closer = _grouping.GuardrailStageCloser()

        with graph.node("model") as span:
            closer.on_start(span)  # type: ignore[arg-type]
        closer.on_end(span)  # type: ignore[arg-type]

        lock.__enter__.assert_not_called()

    def test_starting_a_container_does_not_deadlock(
        self, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A container starts under the lock, and with another container open
        # the closer's on_start takes it again. Its own provider, so a deadlock
        # fails the test instead of hanging the fixture's shutdown.
        provider = TracerProvider()
        monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
        tracer = provider.get_tracer(_OPENINFERENCE)
        other_tools, tools = tracer.start_span("tools"), tracer.start_span("tools")
        current = [other_tools]
        enable_guardrail_grouping(provider, lambda: current[0])
        _pii()
        with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
            pass  # stays open under other_tools
        current[0] = tools

        def check() -> None:
            with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
                pass

        worker = threading.Thread(target=check, daemon=True)
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert len(_grouping._state.containers) == 2
        tools.end()
        other_tools.end()
        provider.shutdown()

    def test_a_new_stage_ends_the_previous_one_without_the_closer(
        self, graph: _Graph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Containers from a provider the closer isn't attached to, so it never
        # sees their start (as when another manager's shutdown detached it).
        hidden = TracerProvider()
        finished = InMemorySpanExporter()
        hidden.add_span_processor(SimpleSpanProcessor(finished))
        monkeypatch.setattr(trace, "get_tracer", hidden.get_tracer)
        _pii()

        with graph.node("tools"):
            with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
                pass
            with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.POST):
                pass

            ended = [s.name for s in finished.get_finished_spans()]
            assert ended == ["Tool input guardrail check"]
        graph.finish()
        hidden.shutdown()

    def test_tool_only_guardrails_turn_grouping_on(
        self, exporter: InMemorySpanExporter
    ) -> None:
        manager = MagicMock()
        enable_guardrail_grouping(manager, lambda: None)
        _rule(GuardrailExecutionStage.PRE)

        manager.add_span_processor.assert_called_once()
        assert _grouping._state.enabled

    def test_grouping_stays_off_without_guardrail_middlewares(
        self, exporter: InMemorySpanExporter
    ) -> None:
        manager = MagicMock()
        enable_guardrail_grouping(manager, lambda: None)

        manager.add_span_processor.assert_not_called()
        assert not _grouping._state.enabled

    async def test_only_guardrail_node_spans_are_hidden(self, graph: _Graph) -> None:
        pii = _pii()
        hook = _hook(pii, "_before_model")
        node_name = f"{hook.name}.before_model"
        user_span = trace.get_tracer("my_app").start_span(node_name)
        user_span.end()

        with (
            patch.object(pii, "_evaluate_guardrail", return_value=_FAILED),
            graph.node(node_name),
        ):
            await hook.abefore_model(_STATE, MagicMock())
        spans = graph.finish()

        hidden = [s for s in spans if not is_exported(s)]
        assert [s.instrumentation_scope.name for s in hidden] == [_OPENINFERENCE]  # type: ignore[union-attr]
        assert is_exported(_named(spans, "LLM input guardrail check")[0])

    def test_a_node_that_cannot_be_grouped_stays_visible(self, graph: _Graph) -> None:
        hook = _hook(_pii(), "_before_model")
        node_name = f"{hook.name}.before_model"
        # A sampler that keeps the nodes although one's parent was dropped.
        provider = TracerProvider(sampler=ALWAYS_ON)
        tracer = provider.get_tracer(_OPENINFERENCE)
        unsampled = NonRecordingSpan(
            SpanContext(
                trace_id=1, span_id=2, is_remote=False, trace_flags=TraceFlags(0)
            )
        )
        under_unsampled = tracer.start_span(
            node_name, context=trace.set_span_in_context(unsampled)
        )
        without_parent = tracer.start_span(node_name)

        assert is_exported(under_unsampled)  # type: ignore[arg-type]
        assert is_exported(without_parent)  # type: ignore[arg-type]
        under_unsampled.end()
        without_parent.end()
        provider.shutdown()


class _ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, *args: Any, **kwargs: Any) -> "_ToolCallingModel":
        return self


@tool
def get_weather(city: str) -> str:
    """Weather for a city."""
    return "sunny"


class TestRealAgent:
    """A real create_agent graph traced by OpenInference."""

    @pytest.fixture
    def instrumented(self, exporter: InMemorySpanExporter) -> Iterator[None]:
        provider = exporter.provider  # type: ignore[attr-defined]
        instrumentor = LangChainInstrumentor()
        # Instrumentation is global: re-instrument on this provider, then restore.
        was_instrumented = instrumentor.is_instrumented_by_opentelemetry
        if was_instrumented:
            instrumentor.uninstrument()
        instrumentor.instrument(tracer_provider=provider)
        enable_guardrail_grouping(provider, get_current_span)
        UiPathSpanUtils.register_current_span_ancestors_provider(get_ancestor_spans)
        yield
        instrumentor.uninstrument()
        if was_instrumented:
            instrumentor.instrument()

    def _agent(self, parallel_calls: int) -> Any:
        calls = [
            {"id": f"t{i}", "name": "get_weather", "args": {"city": "London"}}
            for i in range(parallel_calls)
        ]
        model = _ToolCallingModel(
            messages=iter([AIMessage(content="", tool_calls=calls), AIMessage("ok")])
        )
        guardrails = [
            *_pii(
                scopes=[GuardrailScope.LLM, GuardrailScope.TOOL], tools=[get_weather]
            ),
            *UiPathPromptInjectionMiddleware(action=LogAction()),
            *_rule(GuardrailExecutionStage.PRE_AND_POST),
        ]
        return create_agent(model, tools=[get_weather], middleware=guardrails)

    @staticmethod
    def _assert_well_formed(spans: list[ReadableSpan]) -> None:
        by_id = {s.context.span_id: s for s in spans}
        exported = [s for s in spans if is_exported(s)]
        orphans = [
            s.name
            for s in exported
            if s.parent
            and s.parent.span_id in by_id
            and not is_exported(by_id[s.parent.span_id])
        ]
        assert orphans == []
        assert _grouping._state.containers == {}

    async def test_parallel_tool_calls_group_per_tools_span(
        self, exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            await self._agent(parallel_calls=2).ainvoke(
                {"messages": [HumanMessage("a@b.com weather")]}
            )
        spans = list(exporter.get_finished_spans())

        self._assert_well_formed(spans)
        assert len(_named(spans, "LLM input guardrail check")) == 2
        assert len(_named(spans, "LLM output guardrail check")) == 1
        assert len(_named(spans, "Tool input guardrail check")) == 2
        assert len(_named(spans, "Tool output guardrail check")) == 2
        for container in _named(spans, "LLM input guardrail check"):
            assert _children(spans, container) == ["Log", "Log"]

    async def test_overlapping_checks_on_parallel_tool_calls_stay_in_their_container(
        self, exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        def slow_check(*args: Any, **kwargs: Any) -> GuardrailValidationResult:
            time.sleep(0.05)  # long enough for the calls' checks to overlap
            return _FAILED

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin, "_evaluate_guardrail", slow_check
        ):
            await self._agent(parallel_calls=3).ainvoke(
                {"messages": [HumanMessage("a@b.com weather")]}
            )
        spans = list(exporter.get_finished_spans())

        # Each tool call runs in its own tools span, so each has its own container,
        # and every check ends inside it.
        containers = _named(spans, "Tool input guardrail check")
        assert len(containers) == 3
        assert len({c.parent.span_id for c in containers if c.parent}) == 3
        for container in containers:
            children = [
                s
                for s in spans
                if s.parent and s.parent.span_id == container.context.span_id
            ]
            assert children
            assert all(
                c.end_time is not None
                and container.end_time is not None
                and c.end_time <= container.end_time
                for c in children
            )

    async def test_concurrent_runs_keep_their_own_containers(
        self, exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            await asyncio.gather(
                self._agent(parallel_calls=1).ainvoke(
                    {"messages": [HumanMessage("a@b.com")]}
                ),
                self._agent(parallel_calls=1).ainvoke(
                    {"messages": [HumanMessage("a@b.com")]}
                ),
            )
        spans = list(exporter.get_finished_spans())

        self._assert_well_formed(spans)
        containers = _named(spans, "LLM input guardrail check")
        assert len(containers) == 4
        assert all(_children(spans, c) for c in containers)
