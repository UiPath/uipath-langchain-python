"""Tests for Tool guardrails subgraph construction."""

import types
from typing import Sequence
from unittest.mock import MagicMock

from uipath.core.guardrails import BaseGuardrail, GuardrailSelector
from uipath.platform.guardrails import GuardrailScope

import uipath_langchain.agent.react.guardrails.guardrails_subgraph as mod
from tests.agent.guardrails.test_guardrail_utils import (
    FakeStateGraph,
    fake_action,
    fake_factory,
)
from uipath_langchain.agent.guardrails.actions import GuardrailAction


class TestToolGuardrailsSubgraph:
    def test_no_applicable_guardrails_returns_original_node(self):
        tool_name = "test_tool"
        inner = (tool_name, lambda s: s)
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = []

        # Case with empty guardrails
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

    def test_two_guardrails_build_chains_pre_and_post(self, monkeypatch):
        """Two guardrails should create reverse-ordered pre/post chains with failure edges."""
        monkeypatch.setattr(mod, "StateGraph", FakeStateGraph)
        monkeypatch.setattr(mod, "START", "START")
        monkeypatch.setattr(mod, "END", "END")
        # Use fake factory to control eval node names
        monkeypatch.setattr(mod, "create_tool_guardrail_node", fake_factory("eval"))

        tool_name = "test_tool"

        # Guardrails g1 (first), g2 (second); builder processes last first
        guardrail1 = MagicMock()
        guardrail1.name = "guardrail1"
        guardrail1.selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL], match_names=[tool_name]
        )

        guardrail2 = MagicMock()
        guardrail2.name = "guardrail2"
        guardrail2.selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL], match_names=[tool_name]
        )

        a1 = fake_action("log")
        a2 = fake_action("block")
        guardrails = [(guardrail1, a1), (guardrail2, a2)]

        inner = (tool_name, lambda s: s)
        compiled = mod.create_tool_guardrails_subgraph(
            tool_node=inner,
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
            (block_pre_g2, tool_name),
            # Inner to post-execution chain
            (tool_name, post_g1),
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
            tool_name,
        ]:
            assert name in node_names
