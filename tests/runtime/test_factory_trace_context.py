"""Coded agents: LLM Gateway trace context and client-side LLM spans.

Covers the two halves of the factory wiring that let the LLM Gateway publish
its governance / guardrail / "Model run" spans into a coded agent's trace:

* every chat model in the graph sends the trace-context headers, including
  models constructed directly rather than through ``get_chat_model()``;
* the coded runtime's graph run carries
  ``uipath_langchain_client.UIPATH_SKIP_CLIENT_LLM_SPANS_KEY`` in its metadata,
  so UiPath chat models (``UiPathBaseChatModel``) don't create the client-side
  LLM span the Gateway's "Model run" replaces. Direct-provider models, which
  nothing else traces, keep theirs.

The Gateway fakes here are real ``UiPathBaseChatModel`` subclasses with canned
replies, so they take the same path as the UiPath models.
"""

from collections.abc import Iterator
from pathlib import Path
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
from uipath.llm_client.utils.exceptions import UiPathError
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)
from uipath.runtime import UiPathRuntimeContext
from uipath_langchain_client import UIPATH_SKIP_CLIENT_LLM_SPANS_KEY
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.settings import ApiType, RoutingMode, UiPathAPIConfig

from tests.settings import agent_hub_dummy_settings
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime

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


class _DirectModel(FakeListChatModel):
    """A model that calls its provider directly: not a ``UiPathBaseChatModel``."""


_GATEWAY_API_CONFIG = UiPathAPIConfig(
    api_type=ApiType.COMPLETIONS,
    routing_mode=RoutingMode.NORMALIZED,
    freeze_base_url=True,
)


class _GatewayModel(UiPathBaseChatModel, _RecordingModel):
    """An LLM Gateway model with canned replies."""

    api_config: UiPathAPIConfig = _GATEWAY_API_CONFIG


class _FailingGatewayModel(UiPathBaseChatModel, _FailingModel):
    api_config: UiPathAPIConfig = _GATEWAY_API_CONFIG


def _gateway_model(**kwargs: Any) -> _GatewayModel:
    return _GatewayModel(
        model_name="gpt-4o", settings=agent_hub_dummy_settings, **kwargs
    )


class _Recorder(SpanProcessor):
    """Captures span names at start (what live tracking sees) and at end."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[ReadableSpan] = []

    def on_start(self, span: Span, parent_context: Any = None) -> None:
        self.started.append(span.name)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.append(span)


def _graph(*llms: FakeListChatModel) -> Any:
    async def model(state: _State) -> _State:
        text = state["text"]
        for llm in llms:
            text = str((await llm.ainvoke(text)).content)
        return {"text": text}

    g = StateGraph(_State)
    g.add_node("model", model)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    return g.compile()


@pytest.fixture
def factory(tmp_path: Path) -> UiPathLangGraphRuntimeFactory:
    ctx = UiPathRuntimeContext(
        runtime_dir=str(tmp_path), state_file=str(tmp_path / "state.db")
    )
    return UiPathLangGraphRuntimeFactory(ctx)


@pytest.fixture
def recorder() -> Iterator[_Recorder]:
    """Re-instrument OpenInference against a local provider, as the factory does."""
    instrumentor = LangChainInstrumentor()
    instrumentor.uninstrument()
    provider = TracerProvider()
    rec = _Recorder()
    provider.add_span_processor(rec)
    instrumentor.instrument(tracer_provider=provider)
    yield rec
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    for recorded in (_seen_headers, _own_span, _current_span):
        recorded.clear()
    set_dynamic_request_headers({})
    yield
    set_dynamic_request_headers({})


async def _run(factory: UiPathLangGraphRuntimeFactory, *llms: FakeListChatModel) -> Any:
    """Run the graph with exactly the config the coded runtime builds."""
    runtime = await factory._create_runtime_instance(
        compiled_graph=_graph(*llms), runtime_id="rt", entrypoint="agent"
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
    async def test_never_starts_client_span_for_gateway_llm(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        await _run(factory, _gateway_model(responses=["ok"]))

        # Not just dropped at export: never started, so live tracking can't
        # leave it behind as "running".
        assert "_GatewayModel" not in recorder.started
        assert "_GatewayModel" not in [s.name for s in recorder.ended]
        # The rest of the graph is still traced.
        assert "model" in recorder.started
        assert "LangGraph" in recorder.started

    async def test_keeps_client_span_for_direct_provider_llm(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        # Not a Gateway model: no "Model run" replaces it, so the span stays.
        await _run(factory, _DirectModel(responses=["ok"]))

        assert "_DirectModel" in recorder.started
        llm_span = next(s for s in recorder.ended if s.name == "_DirectModel")
        assert llm_span.attributes is not None
        assert llm_span.attributes["openinference.span.kind"] == "LLM"

    async def test_mixed_graph_skips_only_the_gateway_llm(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        await _run(
            factory,
            _gateway_model(responses=["from gateway"]),
            _DirectModel(responses=["from provider"]),
        )

        assert "_GatewayModel" not in recorder.started
        assert "_DirectModel" in recorder.started

    async def test_keeps_parent_for_traceparent_and_governance(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        await _run(factory, _gateway_model(responses=["ok"]))

        # The traceparent span (and governance's enclosing-span fallback) is
        # the graph node that made the call.
        assert _current_span[0] is not None
        assert _current_span[0].name == "model"
        # Governance's own-span lookup finds nothing and falls back.
        assert _own_span == [None]

    async def test_direct_provider_llm_keeps_its_own_span_for_governance(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        await _run(factory, _RecordingModel(responses=["ok"]))

        assert _own_span[0] is not None
        assert _own_span[0].name == "_RecordingModel"

    async def test_model_error_still_propagates(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        llm = _FailingGatewayModel(
            model_name="gpt-4o", settings=agent_hub_dummy_settings, responses=["unused"]
        )
        # Gateway models wrap provider errors in UiPathError.
        with pytest.raises(UiPathError, match="gateway said no"):
            await _run(factory, llm)

        node = next(s for s in recorder.ended if s.name == "model")
        assert not node.status.is_ok
        assert "_FailingGatewayModel" not in recorder.started

    async def test_runtime_built_without_the_factory_keeps_client_llm_spans(
        self, factory: UiPathLangGraphRuntimeFactory, recorder: _Recorder
    ) -> None:
        # Low-code's AgentsRuntimeFactory builds its runtime without
        # _create_runtime_instance, so its run has no skip metadata and keeps
        # the span, even in a process where a coded runtime was created.
        await _run(factory, _gateway_model(responses=["ok"]))
        runtime = UiPathLangGraphRuntime(_graph(_gateway_model(responses=["ok"])))

        await runtime.graph.ainvoke({"text": "hi"}, config=runtime._get_graph_config())

        assert recorder.started.count("_GatewayModel") == 1

    async def test_coded_runtime_config_carries_the_skip_key(
        self, factory: UiPathLangGraphRuntimeFactory
    ) -> None:
        runtime = await factory._create_runtime_instance(
            compiled_graph=_graph(), runtime_id="rt", entrypoint="agent"
        )
        try:
            config = runtime.delegate._get_graph_config()  # type: ignore[attr-defined]
        finally:
            await factory.dispose()

        assert config["metadata"] == {UIPATH_SKIP_CLIENT_LLM_SPANS_KEY: True}
