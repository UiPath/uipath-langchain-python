"""Tests for guardrails subgraph construction."""

import types

import pytest
from uipath.platform.guardrails import GuardrailScope

import uipath_langchain.agent.guardrails.guardrails_subgraph as mod


class FakeStateGraph:
    def __init__(self, _state_type):
        self.added_nodes = []
        self.added_edges = []

    def add_node(self, name, node):
        self.added_nodes.append((name, node))

    def add_edge(self, src, dst):
        self.added_edges.append((src, dst))

    def compile(self):
        # Return a simple object we can inspect if needed
        return types.SimpleNamespace(nodes=self.added_nodes, edges=self.added_edges)


def _fake_action(fail_prefix):
    class _Action:
        def action_node(self, guardrail, scope, execution_stage):
            name = f"{fail_prefix}_{execution_stage}_{guardrail.name}"
            return name, lambda s: s

    return _Action()


def _fake_factory(eval_prefix):
    def _factory(guardrail, execution_stage, success_node, failure_node):
        name = f"{eval_prefix}_{execution_stage}_{guardrail.name}"
        return name, (lambda s: s)  # node function not invoked in this test

    return _factory


class TestGuardrailsSubgraph:
    def test_no_guardrails_creates_simple_edges(self, monkeypatch):
        """When no guardrails, edges should be: START -> inner -> END."""
        monkeypatch.setattr(mod, "StateGraph", FakeStateGraph)
        monkeypatch.setattr(mod, "START", "START")
        monkeypatch.setattr(mod, "END", "END")

        inner = ("inner", lambda s: s)
        compiled = mod.create_guardrails_subgraph(
            main_inner_node=inner,
            guardrails=None,
            scope=GuardrailScope.LLM,
            node_factory=_fake_factory("eval"),
        )

        # Expect only inner node added and two edges
        assert ("inner", inner[1]) in compiled.nodes
        assert set(compiled.edges) == {("START", "inner"), ("inner", "END")}

    def test_two_guardrails_build_chains_pre_and_post(self, monkeypatch):
        """Two guardrails should create reverse-ordered pre/post chains with failure edges."""
        monkeypatch.setattr(mod, "StateGraph", FakeStateGraph)
        monkeypatch.setattr(mod, "START", "START")
        monkeypatch.setattr(mod, "END", "END")

        # Guardrails g1 (first), g2 (second); builder processes last first
        guardrail1 = types.SimpleNamespace(name="guardrail1")
        guardrail2 = types.SimpleNamespace(name="guardrail2")
        a1 = _fake_action("log")
        a2 = _fake_action("block")
        guardrails = [(guardrail1, a1), (guardrail2, a2)]

        inner = ("inner", lambda s: s)
        compiled = mod.create_guardrails_subgraph(
            main_inner_node=inner,
            guardrails=guardrails,
            scope=GuardrailScope.LLM,
            node_factory=_fake_factory("eval"),
        )

        # Expected node names
        pre_g1 = "eval_PreExecution_guardrail1"
        log_pre_g1 = "log_PreExecution_guardrail1"
        pre_g2 = "eval_PreExecution_guardrail2"
        block_pre_g2 = "block_PreExecution_guardrail2"
        post_g1 = "eval_PostExecution_guardrail1"
        log_post_g1 = "log_PostExecution_guardrail1"
        post_g2 = "eval_PostExecution_guardrail2"
        block_post_g2 = "block_PostExecution_guardrail2"

        # Edges (order not guaranteed; compare as a set)
        expected_edges = {
            # Pre-execution chain
            ("START", pre_g1),
            (log_pre_g1, pre_g2),
            (block_pre_g2, "inner"),
            # Inner to post-execution chain
            ("inner", post_g1),
            # Post-execution failure routing to END
            (log_post_g1, post_g2),
            (block_post_g2, "END"),
        }
        assert expected_edges.issubset(set(compiled.edges))

        # Ensure expected nodes are present
        node_names = {name for name, _ in compiled.nodes}
        for name in [
            pre_g1,
            pre_g2,
            post_g1,
            post_g2,
            log_pre_g1,
            block_pre_g2,
            log_post_g1,
            block_post_g2,
            "inner",
        ]:
            assert name in node_names
