"""Tests for agent guardrails subgraph construction."""

from unittest.mock import MagicMock

import pytest
import uipath.platform.guardrails as guardrails_mod
from uipath.platform.guardrails import (
    GuardrailScope,
)

import uipath_langchain.agent.guardrails.guardrails_subgraph as mod
from tests.agent.guardrails.test_guardrail_utils import (
    FakeStateGraph,
    fake_action,
    fake_factory,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage


class TestAgentGuardrailsSubgraph:
    @pytest.mark.parametrize(
        "stage",
        [ExecutionStage.PRE_EXECUTION, ExecutionStage.POST_EXECUTION],
    )
    def test_no_guardrails_edges_for_stage(self, monkeypatch, stage):
        """For both PRE and POST stages with no guardrails, edges are START -> inner -> END."""
        monkeypatch.setattr(mod, "StateGraph", FakeStateGraph)
        monkeypatch.setattr(mod, "START", "START")
        monkeypatch.setattr(mod, "END", "END")

        inner = ("inner", lambda s: s)
        compiled = mod.create_agent_guardrails_subgraph(
            agent_node=inner,
            guardrails=None,
            execution_stage=stage,
        )

        assert ("inner", inner[1]) in compiled.nodes
        assert set(compiled.edges) == {("START", "inner"), ("inner", "END")}

    @pytest.mark.parametrize(
        "stage,expect_pre,expect_post",
        [
            (ExecutionStage.PRE_EXECUTION, True, False),
            (ExecutionStage.POST_EXECUTION, False, True),
        ],
    )
    def test_two_guardrails_build_chain_for_stage(
        self, monkeypatch, stage, expect_pre, expect_post
    ):
        """With two guardrails, builds only the chain for the requested stage."""
        monkeypatch.setattr(mod, "StateGraph", FakeStateGraph)
        monkeypatch.setattr(mod, "START", "START")
        monkeypatch.setattr(mod, "END", "END")
        monkeypatch.setattr(mod, "create_agent_guardrail_node", fake_factory("eval"))

        guardrail1 = MagicMock()
        guardrail1.name = "guardrail1"
        guardrail1.selector = guardrails_mod.GuardrailSelector(
            scopes=[GuardrailScope.AGENT]
        )

        guardrail2 = MagicMock()
        guardrail2.name = "guardrail2"
        guardrail2.selector = guardrails_mod.GuardrailSelector(
            scopes=[GuardrailScope.AGENT]
        )

        a1 = fake_action("log")
        a2 = fake_action("block")
        guardrails = [(guardrail1, a1), (guardrail2, a2)]

        inner = ("inner", lambda s: s)
        compiled = mod.create_agent_guardrails_subgraph(
            agent_node=inner,
            guardrails=guardrails,
            execution_stage=stage,
        )

        expected_edges = set()
        if expect_pre:
            pre_g1 = "eval_pre_execution_guardrail1"
            log_pre_g1 = "log_pre_execution_guardrail1"
            pre_g2 = "eval_pre_execution_guardrail2"
            block_pre_g2 = "block_pre_execution_guardrail2"
            expected_edges |= {
                ("START", pre_g1),
                (log_pre_g1, pre_g2),
                (block_pre_g2, "inner"),
            }
        else:
            expected_edges |= {("START", "inner")}

        if expect_post:
            post_g1 = "eval_post_execution_guardrail1"
            log_post_g1 = "log_post_execution_guardrail1"
            post_g2 = "eval_post_execution_guardrail2"
            block_post_g2 = "block_post_execution_guardrail2"
            expected_edges |= {
                ("inner", post_g1),
                (log_post_g1, post_g2),
                (block_post_g2, "END"),
            }
        else:
            expected_edges |= {("inner", "END")}

        assert expected_edges.issubset(set(compiled.edges))

        node_names = {name for name, _ in compiled.nodes}
        if expect_pre:
            assert any("pre_execution" in name for name in node_names)
        else:
            assert all("pre_execution" not in name for name in node_names)
        if expect_post:
            assert any("post_execution" in name for name in node_names)
        else:
            assert all("post_execution" not in name for name in node_names)
