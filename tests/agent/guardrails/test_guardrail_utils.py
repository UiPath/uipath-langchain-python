"""Shared utilities for guardrail tests."""

from __future__ import annotations

import types
from typing import Any, Callable, Protocol

from uipath.platform.guardrails import BaseGuardrail

from uipath_langchain.agent.guardrails.actions.base_action import (
    GuardrailAction,
    GuardrailActionNode,
)


class _CompiledGraph(Protocol):
    """Protocol for compiled fake graphs used in tests."""

    nodes: list[tuple[str, Any]]
    edges: list[tuple[str, str]]

    async def ainvoke(self, state: Any) -> dict[str, Any]:
        """Invoke the compiled graph."""


class FakeStateGraph:
    """Minimal fake of LangGraph's StateGraph used by guardrail-subgraph unit tests."""

    def __init__(self, _state_type: Any) -> None:
        """Create a fake graph container."""
        self.added_nodes: list[tuple[str, Any]] = []
        self.added_edges: list[tuple[str, str]] = []

    def add_node(self, name: str, node: Any) -> None:
        """Record a node added to the graph."""
        self.added_nodes.append((name, node))

    def add_edge(self, src: str, dst: str) -> None:
        """Record an edge added to the graph."""
        self.added_edges.append((src, dst))

    def compile(self) -> Any:
        """Compile the fake graph into an inspectable object."""
        # Return a simple object we can inspect if needed.
        return types.SimpleNamespace(nodes=self.added_nodes, edges=self.added_edges)


class FakeStateGraphWithAinvoke(FakeStateGraph):
    """Fake graph that exposes an async `ainvoke` on the compiled output.

    Useful for wrapper functions that expect `compiled_graph.ainvoke(...)`
    (e.g., terminate subgraph wrappers).
    """

    def __init__(self, _state_type: Any) -> None:
        """Create a fake graph container, tracking the first-added node as the entry node."""
        super().__init__(_state_type)
        self._main_node_name: str | None = None

    def add_node(self, name: str, node: Any) -> None:
        """Record a node and remember the first-added node name as the main node."""
        if self._main_node_name is None:
            self._main_node_name = name
        super().add_node(name, node)

    def compile(self) -> _CompiledGraph:
        """Compile the fake graph and attach an async `ainvoke` that runs the main node."""
        compiled = super().compile()
        compiled.main_node_name = self._main_node_name

        async def ainvoke(state: Any) -> dict[str, Any]:
            node_map: dict[str, Callable[[Any], dict[str, Any]]] = {
                n: fn for n, fn in compiled.nodes
            }
            main_node_name = compiled.main_node_name
            if main_node_name is None:
                raise RuntimeError("FakeStateGraphWithAinvoke has no nodes to invoke.")
            return node_map[main_node_name](state)

        compiled.ainvoke = ainvoke
        return compiled


def fake_action(fail_prefix: str) -> GuardrailAction:
    class _Action(GuardrailAction):
        def action_node(
            self,
            *,
            guardrail: BaseGuardrail,
            scope,
            execution_stage,
            guarded_component_name: str,
        ) -> GuardrailActionNode:
            name = f"{fail_prefix}_{execution_stage.name.lower()}_{guardrail.name}"
            return name, lambda s: s

    return _Action()


def fake_factory(eval_prefix):
    def _factory(guardrail, execution_stage, success_node, failure_node, **_kwargs):
        name = f"{eval_prefix}_{execution_stage.name.lower()}_{guardrail.name}"
        return name, (lambda s: s)  # node function not invoked in this test

    return _factory
