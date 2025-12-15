"""Tests for guardrails subgraph construction."""

import types
from typing import Sequence
from unittest.mock import MagicMock

from uipath.platform.guardrails import (
    BaseGuardrail,
    GuardrailScope,
)

import uipath_langchain.agent.guardrails.guardrails_subgraph as mod
from uipath_langchain.agent.guardrails.actions.base_action import (
    GuardrailAction,
    GuardrailActionNode,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage


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


def _fake_action(fail_prefix: str) -> GuardrailAction:
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


def _fake_factory(eval_prefix):
    def _factory(guardrail, execution_stage, success_node, failure_node):
        name = f"{eval_prefix}_{execution_stage.name.lower()}_{guardrail.name}"
        return name, (lambda s: s)  # node function not invoked in this test

    return _factory


class TestLlmGuardrailsSubgraph:
    def test_no_applicable_guardrails_returns_original_node(self):
        """If no guardrails match the scope, the original node should be returned."""
        inner = ("inner", lambda s: s)
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = []

        # Should return the inner node function directly, not a compiled graph
        result = mod.create_llm_guardrails_subgraph(
            llm_node=inner, guardrails=guardrails
        )
        assert result == inner[1]

        # Case with None guardrails
        result_none = mod.create_llm_guardrails_subgraph(
            llm_node=inner, guardrails=None
        )
        assert result_none == inner[1]

        # Case with guardrails but none matching LLM scope
        non_matching_guardrail = MagicMock()
        non_matching_guardrail.selector = types.SimpleNamespace(
            scopes=[GuardrailScope.TOOL]
        )
        guardrails_non_match = [(non_matching_guardrail, MagicMock())]

        result_non_match = mod.create_llm_guardrails_subgraph(
            llm_node=inner, guardrails=guardrails_non_match
        )
        assert result_non_match == inner[1]

    def test_two_guardrails_build_chains_pre_and_post(self, monkeypatch):
        """Two guardrails should create reverse-ordered pre/post chains with failure edges."""
        monkeypatch.setattr(mod, "StateGraph", FakeStateGraph)
        monkeypatch.setattr(mod, "START", "START")
        monkeypatch.setattr(mod, "END", "END")
        # Use fake factory to control eval node names
        monkeypatch.setattr(mod, "create_llm_guardrail_node", _fake_factory("eval"))

        # Guardrails g1 (first), g2 (second); builder processes last first
        guardrail1 = MagicMock()
        guardrail1.name = "guardrail1"
        guardrail1.selector = types.SimpleNamespace(scopes=[GuardrailScope.LLM])

        guardrail2 = MagicMock()
        guardrail2.name = "guardrail2"
        guardrail2.selector = types.SimpleNamespace(scopes=[GuardrailScope.LLM])

        a1 = _fake_action("log")
        a2 = _fake_action("block")
        guardrails = [(guardrail1, a1), (guardrail2, a2)]

        inner = ("inner", lambda s: s)
        compiled = mod.create_llm_guardrails_subgraph(
            llm_node=inner,
            guardrails=guardrails,
        )

        # Expected node names
        pre_g1 = "eval_pre_execution_guardrail1"
        log_pre_g1 = "log_pre_execution_guardrail1"
        pre_g2 = "eval_pre_execution_guardrail2"
        block_pre_g2 = "block_pre_execution_guardrail2"
        post_g1 = "eval_post_execution_guardrail1"
        log_post_g1 = "log_post_execution_guardrail1"
        post_g2 = "eval_post_execution_guardrail2"
        block_post_g2 = "block_post_execution_guardrail2"

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


class TestAgentGuardrailsSubgraph:
    def test_no_applicable_guardrails_returns_original_node(self):
        """If no guardrails match the AGENT scope, the original node should be returned."""
        inner = ("inner", lambda s: s)
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = []
        stage = ExecutionStage.PRE_EXECUTION

        # Should return the inner node function directly
        result = mod.create_agent_guardrails_subgraph(
            agent_node=inner, guardrails=guardrails, execution_stage=stage
        )
        assert result == inner[1]

        # Case with None guardrails
        result_none = mod.create_agent_guardrails_subgraph(
            agent_node=inner, guardrails=None, execution_stage=stage
        )
        assert result_none == inner[1]

        # Case with guardrails but none matching AGENT scope
        non_matching_guardrail = MagicMock()
        non_matching_guardrail.selector = types.SimpleNamespace(
            scopes=[GuardrailScope.LLM]
        )
        guardrails_non_match = [(non_matching_guardrail, MagicMock())]

        result_non_match = mod.create_agent_guardrails_subgraph(
            agent_node=inner, guardrails=guardrails_non_match, execution_stage=stage
        )
        assert result_non_match == inner[1]


class TestToolGuardrailsSubgraph:
    def test_no_applicable_guardrails_returns_original_node(self):
        """If no guardrails match the TOOL scope/name, the original node should be returned."""
        tool_name = "test_tool"
        inner = (tool_name, lambda s: s)
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = []

        # Should return the inner node function directly
        result = mod.create_tool_guardrails_subgraph(
            tool_node=inner, guardrails=guardrails
        )
        assert result == inner[1]

        # Case with None guardrails
        result_none = mod.create_tool_guardrails_subgraph(
            tool_node=inner, guardrails=None
        )
        assert result_none == inner[1]

        # Case with guardrails matching TOOL scope but NOT the tool name
        wrong_name_guardrail = MagicMock()
        wrong_name_guardrail.selector = types.SimpleNamespace(
            scopes=[GuardrailScope.TOOL], match_names=["other_tool"]
        )
        guardrails_wrong_name = [(wrong_name_guardrail, MagicMock())]

        result_wrong_name = mod.create_tool_guardrails_subgraph(
            tool_node=inner, guardrails=guardrails_wrong_name
        )
        assert result_wrong_name == inner[1]

        # Case with guardrails matching tool name but NOT TOOL scope (unlikely but good to test)
        wrong_scope_guardrail = MagicMock()
        wrong_scope_guardrail.selector = types.SimpleNamespace(
            scopes=[GuardrailScope.LLM], match_names=[tool_name]
        )
        guardrails_wrong_scope = [(wrong_scope_guardrail, MagicMock())]

        result_wrong_scope = mod.create_tool_guardrails_subgraph(
            tool_node=inner, guardrails=guardrails_wrong_scope
        )
        assert result_wrong_scope == inner[1]
