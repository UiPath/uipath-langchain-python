"""Coded agents: LLM Gateway trace context and client-side LLM spans.

Covers the two halves of the factory wiring that let the LLM Gateway publish
its governance / guardrail / "Model run" spans into a coded agent's trace:

* every chat model in the graph sends the trace-context headers, including
  models constructed directly rather than through ``get_chat_model()``;
* OpenInference no longer opens the client-side LLM span the Gateway's
  "Model run" replaces.
"""

import os
import tempfile
from collections.abc import Iterator
from typing import Any, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.graph import END, START, StateGraph
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)
from uipath.runtime import UiPathRuntimeContext

from uipath_langchain.runtime.factory import (
    UiPathLangGraphRuntimeFactory,
    _skip_client_llm_spans,
)

_TRACE_HEADERS = {
    "x-uipath-traceparent-id": "00-aabbccddeeff00112233445566778899-1122334455667788-01",
    "x-uipath-tracebaggage": "source=agents",
}


class _State(TypedDict):
    text: str


# Module-level: class attributes on a pydantic model would become per-instance fields.
_seen_headers: list[dict[str, str]] = []
_own_span: list[Any] = []
_current_span: list[Any] = []


class _RecordingModel(FakeListChatModel):
    """Records, while generating, what the HTTP client and governance would see."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _seen_headers.append(dict(get_dynamic_request_headers()))
        if run_manager is not None:
            _own_span.append(LangChainInstrumentor().get_span(run_manager.run_id))
        _current_span.append(get_current_span())
        return super()._generate(messages, stop, run_manager, **kwargs)


class _FailingModel(FakeListChatModel):
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("gateway said no")


class _Recorder(SpanProcessor):
    """Captures span names at start (what live tracking sees) and at end."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[ReadableSpan] = []

    def on_start(self, span: Span, parent_context: Any = None) -> None:
        self.started.append(span.name)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.append(span)


def _graph(llm: FakeListChatModel) -> Any:
    async def model(state: _State) -> _State:
        reply = await llm.ainvoke(state["text"])
        return {"text": str(reply.content)}

    g = StateGraph(_State)
    g.add_node("model", model)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    return g.compile()


@pytest.fixture
def factory() -> Iterator[UiPathLangGraphRuntimeFactory]:
    tmpdir = tempfile.mkdtemp()
    ctx = UiPathRuntimeContext(
        runtime_dir=tmpdir, state_file=os.path.join(tmpdir, "state.db")
    )
    yield UiPathLangGraphRuntimeFactory(ctx)


@pytest.fixture
def recorder() -> Iterator[_Recorder]:
    """Re-instrument OpenInference against a local provider, as the factory does."""
    instrumentor = LangChainInstrumentor()
    instrumentor.uninstrument()
    provider = TracerProvider()
    rec = _Recorder()
    provider.add_span_processor(rec)
    instrumentor.instrument(tracer_provider=provider)
    _skip_client_llm_spans(instrumentor)
    yield rec
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    for recorded in (_seen_headers, _own_span, _current_span):
        recorded.clear()
    set_dynamic_request_headers({})
    yield
    set_dynamic_request_headers({})


async def _run(factory: UiPathLangGraphRuntimeFactory, llm: FakeListChatModel) -> Any:
    """Run the graph with exactly the config the coded runtime builds."""
    runtime = await factory._create_runtime_instance(
        compiled_graph=_graph(llm), runtime_id="rt", entrypoint="agent"
    )
    try:
        config = runtime.delegate._get_graph_config()  # type: ignore[attr-defined]
        return await runtime.delegate.graph.ainvoke({"text": "hi"}, config=config)  # type: ignore[attr-defined]
    finally:
        await factory.dispose()


class TestTraceContextHeaders:
    async def test_directly_constructed_model_sends_trace_headers(
        self, factory: UiPathLangGraphRuntimeFactory
    ) -> None:
        # Built directly (like ``UiPathChatAnthropicBedrock(...)``), not via
        # get_chat_model(), so it has no callback of its own.
        llm = _RecordingModel(responses=["ok"])
        assert not llm.callbacks

        with patch(
            "uipath_langchain.chat.chat_model_factory.build_trace_context_headers",
            return_value=dict(_TRACE_HEADERS),
        ):
            await _run(factory, llm)

        assert len(_seen_headers) == 1
        assert _TRACE_HEADERS.items() <= _seen_headers[0].items()


class TestSkipClientLlmSpans:
    async def test_never_starts_client_llm_span(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        await _run(factory, _RecordingModel(responses=["ok"]))

        # Not just dropped at export: never started, so live tracking can't
        # leave it behind as "running".
        assert "_RecordingModel" not in recorder.started
        assert "_RecordingModel" not in [s.name for s in recorder.ended]
        # The rest of the graph is still traced.
        assert "model" in recorder.started
        assert "LangGraph" in recorder.started

    async def test_keeps_parent_for_traceparent_and_governance(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        await _run(factory, _RecordingModel(responses=["ok"]))

        # The traceparent span (and governance's enclosing-span fallback) is
        # the graph node that made the call.
        assert _current_span[0] is not None
        assert _current_span[0].name == "model"
        # Governance's own-span lookup finds nothing and falls back.
        assert _own_span == [None]

    async def test_model_error_still_propagates(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        with pytest.raises(RuntimeError, match="gateway said no"):
            await _run(factory, _FailingModel(responses=["unused"]))

        node = next(s for s in recorder.ended if s.name == "model")
        assert not node.status.is_ok
        assert "_FailingModel" not in recorder.started

    def test_wrapping_is_idempotent(self, recorder: _Recorder) -> None:
        instrumentor = LangChainInstrumentor()
        wrapped = instrumentor._tracer._start_trace  # type: ignore[union-attr]
        _skip_client_llm_spans(instrumentor)
        assert instrumentor._tracer._start_trace is wrapped  # type: ignore[union-attr]

    def test_factory_setup_installs_the_skip(
        self, factory: UiPathLangGraphRuntimeFactory
    ) -> None:
        # Guards the private OpenInference hook we wrap: if an upgrade renames
        # it, this fails instead of client LLM spans silently coming back.
        tracer = LangChainInstrumentor()._tracer
        assert tracer is not None
        assert callable(tracer._start_trace)
        assert getattr(tracer, "_uipath_skip_llm", False) is True
