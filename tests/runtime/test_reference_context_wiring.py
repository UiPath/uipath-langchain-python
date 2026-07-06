"""Tests for ReferenceContext wiring in UiPathLangGraphRuntime."""

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

try:
    from uipath.tracing import ReferenceContext, ReferenceContextAccessor  # type: ignore[attr-defined]
    _reference_context_available = True
except ImportError:
    _reference_context_available = False
    ReferenceContext = None
    ReferenceContextAccessor = None

from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime

pytestmark = pytest.mark.skipif(
    not _reference_context_available,
    reason="installed uipath does not export ReferenceContext",
)


# ---------------------------------------------------------------------------
# Minimal graph fixture
# ---------------------------------------------------------------------------

class _State(TypedDict):
    value: str


def _build_graph() -> Any:
    graph = StateGraph(_State)
    graph.add_node("step", lambda s: {"value": s.get("value", "") + "_done"})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    return graph


def _clear_accessor() -> None:
    ReferenceContextAccessor.set(None)


# ---------------------------------------------------------------------------
# _push_reference_context — unit tests (no graph needed)
# ---------------------------------------------------------------------------

class TestPushReferenceContext:
    def setup_method(self) -> None:
        _clear_accessor()

    def teardown_method(self) -> None:
        _clear_accessor()

    def test_sets_langgraph_entry_when_agent_id_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("UIPATH_AGENT_ID", "550e8400-e29b-41d4-a716-446655440020")
        monkeypatch.delenv("UIPATH_PROCESS_VERSION", raising=False)

        graph = _build_graph().compile()
        runtime = UiPathLangGraphRuntime(graph=graph, runtime_id="t")

        token = runtime._push_reference_context()
        try:
            ctx = ReferenceContextAccessor.get()
            assert ctx is not None
            assert len(ctx) == 1
            assert ctx.entries[0].service_type == "langgraph"
            assert ctx.entries[0].reference_id == "550e8400-e29b-41d4-a716-446655440020"
            assert ctx.entries[0].version is None
        finally:
            ReferenceContextAccessor.reset(token)

    def test_includes_version_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_AGENT_ID", "550e8400-e29b-41d4-a716-446655440020")
        monkeypatch.setenv("UIPATH_PROCESS_VERSION", "3.1.0")

        graph = _build_graph().compile()
        runtime = UiPathLangGraphRuntime(graph=graph, runtime_id="t")

        token = runtime._push_reference_context()
        try:
            ctx = ReferenceContextAccessor.get()
            assert ctx is not None
            assert ctx.entries[0].version == "3.1.0"
        finally:
            ReferenceContextAccessor.reset(token)

    def test_no_entry_when_agent_id_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UIPATH_AGENT_ID", raising=False)
        monkeypatch.delenv("UIPATH_PROCESS_VERSION", raising=False)

        graph = _build_graph().compile()
        runtime = UiPathLangGraphRuntime(graph=graph, runtime_id="t")

        token = runtime._push_reference_context()
        try:
            ctx = ReferenceContextAccessor.get()
            assert ctx is not None
            assert len(ctx) == 0
        finally:
            ReferenceContextAccessor.reset(token)

    def test_stacks_on_top_of_parent_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_AGENT_ID", "550e8400-e29b-41d4-a716-446655440020")
        monkeypatch.delenv("UIPATH_PROCESS_VERSION", raising=False)

        parent = ReferenceContext.Empty.add(
            "agent", "550e8400-e29b-41d4-a716-446655440001", "1.0"
        )
        parent_token = ReferenceContextAccessor.set(parent)

        graph = _build_graph().compile()
        runtime = UiPathLangGraphRuntime(graph=graph, runtime_id="t")

        token = runtime._push_reference_context()
        try:
            ctx = ReferenceContextAccessor.get()
            assert ctx is not None
            assert len(ctx) == 2
            assert ctx.entries[0].service_type == "agent"
            assert ctx.entries[1].service_type == "langgraph"
        finally:
            ReferenceContextAccessor.reset(token)
            ReferenceContextAccessor.reset(parent_token)


# ---------------------------------------------------------------------------
# execute() — context cleared after run
# ---------------------------------------------------------------------------

async def test_context_cleared_after_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_accessor()
    monkeypatch.setenv("UIPATH_AGENT_ID", "550e8400-e29b-41d4-a716-446655440020")
    monkeypatch.delenv("UIPATH_PROCESS_VERSION", raising=False)

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "mem.db")) as memory:
        await memory.setup()
        graph = _build_graph().compile(checkpointer=memory)
        runtime = UiPathLangGraphRuntime(graph=graph, runtime_id="exec-run")
        await runtime.execute(input={"value": "hello"})

    assert ReferenceContextAccessor.get() is None


async def test_context_cleared_after_execute_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_accessor()
    monkeypatch.setenv("UIPATH_AGENT_ID", "550e8400-e29b-41d4-a716-446655440020")

    class _S(TypedDict):
        v: str

    def _boom(s: _S) -> _S:
        raise ValueError("explode")

    g = StateGraph(_S)
    g.add_node("boom", _boom)  # type: ignore[arg-type]
    g.add_edge(START, "boom")
    g.add_edge("boom", END)

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "mem.db")) as memory:
        await memory.setup()
        compiled = g.compile(checkpointer=memory)
        runtime = UiPathLangGraphRuntime(graph=compiled, runtime_id="err-run")
        with pytest.raises(Exception):
            await runtime.execute(input={"v": "x"})

    assert ReferenceContextAccessor.get() is None


# ---------------------------------------------------------------------------
# stream() — context cleared after run
# ---------------------------------------------------------------------------

async def test_context_cleared_after_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_accessor()
    monkeypatch.setenv("UIPATH_AGENT_ID", "550e8400-e29b-41d4-a716-446655440020")
    monkeypatch.delenv("UIPATH_PROCESS_VERSION", raising=False)

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "mem.db")) as memory:
        await memory.setup()
        graph = _build_graph().compile(checkpointer=memory)
        runtime = UiPathLangGraphRuntime(graph=graph, runtime_id="stream-run")
        async for _ in runtime.stream(input={"value": "hi"}):
            pass

    assert ReferenceContextAccessor.get() is None
