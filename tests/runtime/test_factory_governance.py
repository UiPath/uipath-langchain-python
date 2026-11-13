"""Factory-level governance wiring: evaluator -> callbacks plumbing."""

from __future__ import annotations

import os
import tempfile
from typing import Any, TypedDict
from unittest.mock import MagicMock

import pytest
from langgraph.graph import END, START, StateGraph
from uipath.core.adapters import EvaluatorProtocol
from uipath.runtime import UiPathRuntimeContext

from uipath_langchain.chat.chat_model_factory import _TraceContextHeadersCallback
from uipath_langchain.governance import GovernanceCallbackHandler
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory


class _State(TypedDict):
    v: int


def _governance_handlers(runtime: Any) -> list[GovernanceCallbackHandler]:
    return [
        cb
        for cb in runtime.delegate.callbacks
        if isinstance(cb, GovernanceCallbackHandler)
    ]


def _build_graph() -> StateGraph[Any, Any, Any]:
    g = StateGraph(_State)
    g.add_node("noop", lambda s: s)
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    return g


@pytest.fixture
def context() -> UiPathRuntimeContext:
    tmpdir = tempfile.mkdtemp()
    ctx = UiPathRuntimeContext(
        runtime_dir=tmpdir,
        state_file=os.path.join(tmpdir, "state.db"),
    )
    return ctx


@pytest.fixture
def factory(context: UiPathRuntimeContext) -> UiPathLangGraphRuntimeFactory:
    return UiPathLangGraphRuntimeFactory(context)


class TestEvaluatorWiring:
    """Passing ``evaluator`` to ``new_runtime`` should attach a
    :class:`GovernanceCallbackHandler` to the underlying LangGraph
    runtime's callback list. This is the entire surface change — the
    previous adapter / register-on-import path is gone.
    """

    async def test_no_evaluator_means_only_trace_context_callback(
        self, factory: UiPathLangGraphRuntimeFactory
    ) -> None:
        compiled = _build_graph().compile()
        await factory._get_memory()
        runtime = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="rt-1",
            entrypoint="ep",
        )
        # The resumable runtime wraps the langgraph runtime as ``delegate``.
        callbacks = runtime.delegate.callbacks  # type: ignore[attr-defined]
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], _TraceContextHeadersCallback)
        await factory.dispose()

    async def test_evaluator_attaches_governance_handler(
        self, factory: UiPathLangGraphRuntimeFactory
    ) -> None:
        evaluator: EvaluatorProtocol = MagicMock(spec=EvaluatorProtocol)
        compiled = _build_graph().compile()
        await factory._get_memory()  # ensure memory is initialized
        runtime = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="rt-1",
            entrypoint="ep",
            evaluator=evaluator,
        )
        callbacks = runtime.delegate.callbacks  # type: ignore[attr-defined]
        assert any(isinstance(cb, _TraceContextHeadersCallback) for cb in callbacks)
        handlers = _governance_handlers(runtime)
        assert len(handlers) == 1
        handler = handlers[0]
        # Identity / session_id / agent_name come from the factory args.
        assert handler._evaluator is evaluator
        assert handler._agent_name == "ep"
        assert handler._session_id == "rt-1"
        await factory.dispose()

    async def test_handler_built_per_runtime_instance(
        self, factory: UiPathLangGraphRuntimeFactory
    ) -> None:
        """Two factory calls with the same evaluator yield two distinct
        handler instances — each runtime gets its own session_state, so
        concurrent sessions don't share counters."""
        evaluator: EvaluatorProtocol = MagicMock(spec=EvaluatorProtocol)
        compiled = _build_graph().compile()
        await factory._get_memory()
        first = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="rt-a",
            entrypoint="ep",
            evaluator=evaluator,
        )
        second = await factory._create_runtime_instance(
            compiled_graph=compiled,
            runtime_id="rt-b",
            entrypoint="ep",
            evaluator=evaluator,
        )
        h1 = _governance_handlers(first)[0]
        h2 = _governance_handlers(second)[0]
        assert h1 is not h2
        assert h1._session_id == "rt-a"
        assert h2._session_id == "rt-b"
        assert h1._session_state is not h2._session_state
        await factory.dispose()


class TestGovernanceMetadataLabels:
    """The factory advertises the governance metadata taxonomy wire labels
    via ``get_settings()``. Hosts forward ``agent_type`` / ``agent_framework``
    verbatim onto governance audit telemetry, so these must match the
    canonical taxonomy: ``uipath_coded`` for this coded-agent surface and
    ``langchain`` for the adapter framework (``langgraph`` is the project
    marker/runtime flavor, not the framework label).
    """

    async def test_get_settings_advertises_taxonomy_labels(
        self, factory: UiPathLangGraphRuntimeFactory
    ) -> None:
        settings = await factory.get_settings()
        assert settings is not None
        assert settings.agent_type == "uipath_coded"
        assert settings.agent_framework == "langchain"
