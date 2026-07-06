"""Unit tests for UiPathLangGraphRuntime helper methods (no graph execution)."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langgraph.errors import EmptyInputError, GraphRecursionError, InvalidUpdateError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from uipath.runtime import UiPathRuntimeStatus
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.runtime.errors import LangGraphErrorCode, LangGraphRuntimeError
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime


def _make_runtime(**kwargs: Any) -> UiPathLangGraphRuntime:
    graph = MagicMock(spec=CompiledStateGraph)
    graph.nodes = {}
    graph.output_channels = []
    return UiPathLangGraphRuntime(graph=graph, **kwargs)


def _snapshot(
    next_nodes: tuple = (),
    interrupts: tuple = (),
    tasks: list | None = None,
    values: dict | None = None,
) -> MagicMock:
    snap = MagicMock(spec=StateSnapshot)
    snap.next = next_nodes
    snap.interrupts = interrupts
    snap.tasks = tasks or []
    snap.values = values or {}
    return snap


# ---------------------------------------------------------------------------
# _build_node_name
# ---------------------------------------------------------------------------


class TestBuildNodeName:
    def setup_method(self) -> None:
        self.rt = _make_runtime()

    def test_root_graph_returns_node_name(self) -> None:
        assert self.rt._build_node_name((), "agent") == "agent"

    def test_single_subgraph_prepends_subgraph_name(self) -> None:
        assert self.rt._build_node_name(("coder:abc123",), "generate") == "coder:generate"

    def test_nested_subgraphs_build_full_path(self) -> None:
        assert self.rt._build_node_name(("coder:a", "debugger:b"), "analyze") == "coder:debugger:analyze"

    def test_namespace_without_colon_uses_full_segment(self) -> None:
        assert self.rt._build_node_name(("subgraph",), "node") == "subgraph:node"

    def test_empty_string_segments_are_skipped(self) -> None:
        assert self.rt._build_node_name(("",), "node") == "node"

    def test_non_tuple_namespace_falls_back_to_node_name(self) -> None:
        assert self.rt._build_node_name("unexpected", "node") == "node"

    def test_none_namespace_falls_back_to_node_name(self) -> None:
        assert self.rt._build_node_name(None, "node") == "node"


# ---------------------------------------------------------------------------
# _extract_graph_result
# ---------------------------------------------------------------------------


class TestExtractGraphResult:
    def setup_method(self) -> None:
        self.rt = _make_runtime()

    def test_non_dict_returned_unchanged(self) -> None:
        self.rt.graph.output_channels = ["out"]
        assert self.rt._extract_graph_result("string") == "string"

    def test_single_string_channel_extracts_matching_key(self) -> None:
        self.rt.graph.output_channels = "result"
        assert self.rt._extract_graph_result({"result": 42, "other": 1}) == 42

    def test_single_string_channel_missing_returns_full_chunk(self) -> None:
        self.rt.graph.output_channels = "result"
        chunk = {"other": 1}
        assert self.rt._extract_graph_result(chunk) == chunk

    def test_multi_channel_returns_only_present_keys(self) -> None:
        self.rt.graph.output_channels = ["a", "b", "c"]
        assert self.rt._extract_graph_result({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_multi_channel_unwraps_single_key_wrapping_dict(self) -> None:
        self.rt.graph.output_channels = ["x", "y"]
        assert self.rt._extract_graph_result({"node": {"x": 10, "y": 20}}) == {"x": 10, "y": 20}

    def test_tuple_format_unwraps_second_element(self) -> None:
        self.rt.graph.output_channels = "val"
        assert self.rt._extract_graph_result(("ignored", {"val": 99})) == 99

    def test_empty_sequence_channels_returns_chunk_unchanged(self) -> None:
        self.rt.graph.output_channels = []
        chunk = {"a": 1}
        assert self.rt._extract_graph_result(chunk) == chunk


# ---------------------------------------------------------------------------
# create_runtime_error
# ---------------------------------------------------------------------------


class TestCreateRuntimeError:
    def setup_method(self) -> None:
        self.rt = _make_runtime()

    def test_langgraph_runtime_error_returned_unchanged(self) -> None:
        err = LangGraphRuntimeError(LangGraphErrorCode.GRAPH_LOAD_ERROR, "t", "d", UiPathErrorCategory.USER)
        assert self.rt.create_runtime_error(err) is err

    def test_graph_recursion_error_maps_to_graph_load_error(self) -> None:
        result = self.rt.create_runtime_error(GraphRecursionError("too deep"))
        assert result.error_info.code == "LANGGRAPH.GRAPH_LOAD_ERROR"

    def test_invalid_update_error_maps_to_graph_invalid_update(self) -> None:
        result = self.rt.create_runtime_error(InvalidUpdateError("bad update"))
        assert result.error_info.code == "LANGGRAPH.GRAPH_INVALID_UPDATE"

    def test_empty_input_error_maps_to_graph_empty_input(self) -> None:
        result = self.rt.create_runtime_error(EmptyInputError())
        assert result.error_info.code == "LANGGRAPH.GRAPH_EMPTY_INPUT"

    def test_generic_exception_maps_to_execution_error(self) -> None:
        result = self.rt.create_runtime_error(RuntimeError("boom"))
        assert result.error_info.code == "LANGGRAPH.EXECUTION_ERROR"
        assert "boom" in result.error_info.detail


# ---------------------------------------------------------------------------
# _get_graph_config
# ---------------------------------------------------------------------------


class TestGetGraphConfig:
    def test_thread_id_set_from_runtime_id(self) -> None:
        rt = _make_runtime(runtime_id="my-run")
        assert rt._get_graph_config()["configurable"]["thread_id"] == "my-run"

    def test_no_recursion_limit_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGCHAIN_RECURSION_LIMIT", raising=False)
        assert "recursion_limit" not in _make_runtime()._get_graph_config()

    def test_recursion_limit_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGCHAIN_RECURSION_LIMIT", "50")
        assert _make_runtime()._get_graph_config()["recursion_limit"] == 50

    def test_no_max_concurrency_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGCHAIN_MAX_CONCURRENCY", raising=False)
        assert "max_concurrency" not in _make_runtime()._get_graph_config()

    def test_max_concurrency_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGCHAIN_MAX_CONCURRENCY", "4")
        assert _make_runtime()._get_graph_config()["max_concurrency"] == 4


# ---------------------------------------------------------------------------
# _detect_middleware_nodes / _is_middleware_node
# ---------------------------------------------------------------------------


class TestMiddlewareDetection:
    def _make_with_nodes(self, node_names: list[str]) -> UiPathLangGraphRuntime:
        graph = MagicMock(spec=CompiledStateGraph)
        graph.nodes = {n: MagicMock() for n in node_names}
        graph.output_channels = []
        return UiPathLangGraphRuntime(graph=graph)

    def test_node_with_dot_and_middleware_keyword_detected(self) -> None:
        rt = self._make_with_nodes(["GuardrailsMiddleware.pre", "agent"])
        assert rt._is_middleware_node("GuardrailsMiddleware.pre")

    def test_regular_node_not_detected(self) -> None:
        rt = self._make_with_nodes(["GuardrailsMiddleware.pre", "agent"])
        assert not rt._is_middleware_node("agent")

    def test_node_with_dot_but_no_middleware_keyword_not_detected(self) -> None:
        rt = self._make_with_nodes(["some.node"])
        assert not rt._is_middleware_node("some.node")

    def test_middleware_keyword_without_dot_not_detected(self) -> None:
        rt = self._make_with_nodes(["MiddlewareNode"])
        assert not rt._is_middleware_node("MiddlewareNode")

    def test_multiple_middleware_hooks_all_detected(self) -> None:
        rt = self._make_with_nodes(["FooMiddleware.pre", "FooMiddleware.post", "tools"])
        assert rt._is_middleware_node("FooMiddleware.pre")
        assert rt._is_middleware_node("FooMiddleware.post")
        assert not rt._is_middleware_node("tools")


# ---------------------------------------------------------------------------
# _is_interrupted
# ---------------------------------------------------------------------------


class TestIsInterrupted:
    def setup_method(self) -> None:
        self.rt = _make_runtime()

    def test_true_when_next_nodes_present(self) -> None:
        assert self.rt._is_interrupted(_snapshot(next_nodes=("step",)))

    def test_true_when_dynamic_interrupts_present(self) -> None:
        assert self.rt._is_interrupted(_snapshot(interrupts=(MagicMock(),)))

    def test_false_when_both_empty(self) -> None:
        assert not self.rt._is_interrupted(_snapshot())


# ---------------------------------------------------------------------------
# _create_success_result
# ---------------------------------------------------------------------------


class TestCreateSuccessResult:
    def setup_method(self) -> None:
        self.rt = _make_runtime()

    def test_status_is_successful(self) -> None:
        assert self.rt._create_success_result({"k": "v"}).status == UiPathRuntimeStatus.SUCCESSFUL

    def test_output_passed_through(self) -> None:
        assert self.rt._create_success_result({"key": "val"}).output == {"key": "val"}

    def test_none_output_becomes_empty_dict(self) -> None:
        assert self.rt._create_success_result(None).output == {}


# ---------------------------------------------------------------------------
# _create_breakpoint_result
# ---------------------------------------------------------------------------


class TestCreateBreakpointResult:
    def setup_method(self) -> None:
        self.rt = _make_runtime()

    def test_before_breakpoint_uses_next_node_name(self) -> None:
        snap = _snapshot(next_nodes=("step_b",), values={"k": "v"})
        result = self.rt._create_breakpoint_result(snap)
        assert result.breakpoint_type == "before"
        assert "step_b" in result.breakpoint_node
        assert result.next_nodes == ["step_b"]

    def test_after_breakpoint_uses_last_task_name(self) -> None:
        task = MagicMock()
        task.name = "last_step"
        snap = _snapshot(next_nodes=(), tasks=[task], values={})
        result = self.rt._create_breakpoint_result(snap)
        assert result.breakpoint_type == "after"
        assert result.breakpoint_node == "last_step"

    def test_after_breakpoint_with_no_tasks_is_unknown(self) -> None:
        snap = _snapshot(next_nodes=(), tasks=[], values={})
        result = self.rt._create_breakpoint_result(snap)
        assert result.breakpoint_type == "after"
        assert result.breakpoint_node == "unknown"

    def test_current_state_reflects_snapshot_values(self) -> None:
        snap = _snapshot(next_nodes=("n",), values={"out": "done"})
        result = self.rt._create_breakpoint_result(snap)
        assert result.current_state == {"out": "done"}


# ---------------------------------------------------------------------------
# ImportError fallback — _NoopReferenceContextAccessor
# ---------------------------------------------------------------------------


class TestImportErrorFallback:
    """Cover the _NoopReferenceContextAccessor shim and the ReferenceContext=None
    guard inside _push_reference_context."""

    def test_noop_accessor_get_returns_none(self) -> None:
        from uipath_langchain.runtime.runtime import _NoopReferenceContextAccessor

        assert _NoopReferenceContextAccessor.get() is None

    def test_noop_accessor_set_returns_token(self) -> None:
        import contextvars

        from uipath_langchain.runtime.runtime import _NoopReferenceContextAccessor

        token = _NoopReferenceContextAccessor.set("anything")
        assert isinstance(token, contextvars.Token)

    def test_noop_accessor_reset_does_not_raise(self) -> None:
        from uipath_langchain.runtime.runtime import _NoopReferenceContextAccessor

        token = _NoopReferenceContextAccessor.set(None)
        _NoopReferenceContextAccessor.reset(token)  # must not raise

    def test_push_returns_token_when_reference_context_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import contextvars

        import uipath_langchain.runtime.runtime as rt_mod
        from uipath_langchain.runtime.runtime import _NoopReferenceContextAccessor

        monkeypatch.setattr(rt_mod, "ReferenceContext", None)
        monkeypatch.setattr(rt_mod, "ReferenceContextAccessor", _NoopReferenceContextAccessor)

        rt = _make_runtime()
        token = rt._push_reference_context()
        assert isinstance(token, contextvars.Token)
        # cleanup must not raise
        _NoopReferenceContextAccessor.reset(token)

    def test_push_does_not_set_real_accessor_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from uipath.tracing import ReferenceContextAccessor

        import uipath_langchain.runtime.runtime as rt_mod
        from uipath_langchain.runtime.runtime import _NoopReferenceContextAccessor

        monkeypatch.setattr(rt_mod, "ReferenceContext", None)
        monkeypatch.setattr(rt_mod, "ReferenceContextAccessor", _NoopReferenceContextAccessor)

        rt = _make_runtime()
        before = ReferenceContextAccessor.get()
        token = rt._push_reference_context()
        after = ReferenceContextAccessor.get()
        _NoopReferenceContextAccessor.reset(token)

        assert before == after
