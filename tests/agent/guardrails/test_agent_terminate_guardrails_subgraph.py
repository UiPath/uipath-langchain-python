"""Tests for terminate-node (agent-scope) guardrails wiring.

The terminate node is no longer wrapped in a compiled subgraph. Instead, guardrail nodes
are attached at the parent graph level after TERMINATE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from _pytest.monkeypatch import MonkeyPatch
from uipath.core.guardrails import BaseGuardrail, GuardrailSelector
from uipath.platform.guardrails import GuardrailScope

import uipath_langchain.agent.react.guardrails.guardrails_subgraph as mod
from tests.agent.guardrails.test_guardrail_utils import (
    FakeStateGraph,
    fake_action,
    fake_factory,
)
from uipath_langchain.agent.guardrails.actions import GuardrailAction


@dataclass(frozen=True, slots=True)
class _DummyGuardrail:
    """Minimal guardrail stand-in for unit tests."""

    name: str
    selector: GuardrailSelector


class TestAgentTerminateGuardrailsSubgraph:
    def test_no_applicable_guardrails_returns_next_node(self) -> None:
        """If no guardrails match the AGENT scope, routing should go to the next node."""
        graph = FakeStateGraph(object)
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = []

        def terminate_fn(_state: object) -> dict[str, int]:
            return {"done": 1}

        # Case with empty guardrails
        result = mod.attach_post_agent_guardrails(
            builder=graph,
            terminate_node=terminate_fn,
            guardrails=guardrails,
            terminate_node_name="terminate",
            output_node_name="guarded-terminate",
        )
        assert result == "terminate"

        # Case with None guardrails
        result_none = mod.attach_post_agent_guardrails(
            builder=graph,
            terminate_node=terminate_fn,
            guardrails=None,
            terminate_node_name="terminate",
            output_node_name="guarded-terminate",
        )
        assert result_none == "terminate"

        # Case with guardrails but none matching AGENT scope
        non_matching_guardrail = _DummyGuardrail(
            name="llm_guardrail",
            selector=GuardrailSelector(scopes=[GuardrailScope.LLM]),
        )
        guardrails_non_match = [(non_matching_guardrail, fake_action("noop"))]

        result_non_match = mod.attach_post_agent_guardrails(
            builder=graph,
            terminate_node=terminate_fn,
            guardrails=guardrails_non_match,
            terminate_node_name="terminate",
            output_node_name="guarded-terminate",
        )
        assert result_non_match == "terminate"

    def test_two_guardrails_build_post_chain(self, monkeypatch: MonkeyPatch) -> None:
        """Two AGENT guardrails should create a POST_EXECUTION chain with failure edges."""
        monkeypatch.setattr(
            mod, "create_agent_terminate_guardrail_node", fake_factory("eval")
        )

        guardrail1 = _DummyGuardrail(
            name="guardrail1",
            selector=GuardrailSelector(scopes=[GuardrailScope.AGENT]),
        )
        guardrail2 = _DummyGuardrail(
            name="guardrail2",
            selector=GuardrailSelector(scopes=[GuardrailScope.AGENT]),
        )
        non_matching = _DummyGuardrail(
            name="llm_guardrail",
            selector=GuardrailSelector(scopes=[GuardrailScope.LLM]),
        )

        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = [
            (guardrail1, fake_action("log")),
            (guardrail2, fake_action("block")),
            (non_matching, fake_action("noop")),
        ]

        def terminate_fn(_state: object) -> dict[str, int]:
            return {"done": 1}

        graph = FakeStateGraph(object)
        first_node = mod.attach_post_agent_guardrails(
            builder=graph,
            terminate_node=terminate_fn,
            guardrails=guardrails,
            terminate_node_name="terminate",
            output_node_name="guarded-terminate",
        )
        assert first_node == "guarded-terminate"

        post_g1 = "eval_post_execution_guardrail1"
        log_post_g1 = "log_post_execution_guardrail1"
        post_g2 = "eval_post_execution_guardrail2"
        block_post_g2 = "block_post_execution_guardrail2"

        expected_edges = {
            ("terminate", post_g1),
            (log_post_g1, post_g2),
            (block_post_g2, "guarded-terminate"),
        }
        assert expected_edges.issubset(set(graph.added_edges))

        node_names = {name for name, _ in graph.added_nodes}
        for name in [
            "terminate",
            post_g1,
            post_g2,
            log_post_g1,
            block_post_g2,
            "guarded-terminate",
        ]:
            assert name in node_names
        assert "eval_post_execution_llm_guardrail" not in node_names
        assert "noop_post_execution_llm_guardrail" not in node_names
