"""Tests for terminate-node (agent-scope) guardrails subgraph construction."""

from __future__ import annotations

import types
from typing import Any, Sequence
from unittest.mock import MagicMock

from _pytest.monkeypatch import MonkeyPatch
from uipath.core.guardrails import BaseGuardrail, GuardrailSelector
from uipath.platform.guardrails import GuardrailScope

import uipath_langchain.agent.react.guardrails.guardrails_subgraph as mod
from tests.agent.guardrails.test_guardrail_utils import (
    FakeStateGraphWithAinvoke,
    fake_action,
    fake_factory,
)
from uipath_langchain.agent.guardrails.actions import GuardrailAction


class TestAgentTerminateGuardrailsSubgraph:
    def test_no_applicable_guardrails_returns_original_node(self):
        """If no guardrails match the AGENT scope, the original node should be returned."""
        inner = ("inner", lambda s: s)
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = []

        # Case with empty guardrails
        result = mod.create_agent_terminate_guardrails_subgraph(
            terminate_node=inner, guardrails=guardrails
        )
        assert result == inner[1]

        # Case with None guardrails
        result_none = mod.create_agent_terminate_guardrails_subgraph(
            terminate_node=inner, guardrails=None
        )
        assert result_none == inner[1]

        # Case with guardrails but none matching AGENT scope
        non_matching_guardrail = MagicMock()
        non_matching_guardrail.selector = GuardrailSelector(scopes=[GuardrailScope.LLM])
        guardrails_non_match = [(non_matching_guardrail, MagicMock())]

        result_non_match = mod.create_agent_terminate_guardrails_subgraph(
            terminate_node=inner, guardrails=guardrails_non_match
        )
        assert result_non_match == inner[1]

    async def test_two_guardrails_build_post_chain_and_return_result(
        self, monkeypatch: MonkeyPatch
    ):
        """Two AGENT guardrails should create a POST_EXECUTION chain and returns terminate result."""

        monkeypatch.setattr(mod, "StateGraph", FakeStateGraphWithAinvoke)
        monkeypatch.setattr(mod, "START", "START")
        monkeypatch.setattr(mod, "END", "END")
        monkeypatch.setattr(
            mod, "create_agent_terminate_guardrail_node", fake_factory("eval")
        )

        captured: dict[str, Any] = {}
        original_create = mod._create_guardrails_subgraph

        def _capture_create(*args: Any, **kwargs: Any) -> Any:
            compiled = original_create(*args, **kwargs)
            captured["compiled"] = compiled
            return compiled

        monkeypatch.setattr(mod, "_create_guardrails_subgraph", _capture_create)

        guardrail1 = MagicMock()
        guardrail1.name = "guardrail1"
        guardrail1.selector = GuardrailSelector(scopes=[GuardrailScope.AGENT])

        guardrail2 = MagicMock()
        guardrail2.name = "guardrail2"
        guardrail2.selector = GuardrailSelector(scopes=[GuardrailScope.AGENT])

        non_matching = MagicMock()
        non_matching.name = "llm_guardrail"
        non_matching.selector = GuardrailSelector(scopes=[GuardrailScope.LLM])

        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] = [
            (guardrail1, fake_action("log")),
            (guardrail2, fake_action("block")),
            (non_matching, fake_action("noop")),
        ]

        expected_result: dict[str, Any] = {"done": 1}

        def terminate_fn(_state: Any) -> dict[str, Any]:
            return expected_result

        run_terminate = mod.create_agent_terminate_guardrails_subgraph(
            terminate_node=("terminate", terminate_fn),
            guardrails=guardrails,
        )

        result = await run_terminate(types.SimpleNamespace(messages=["m"]))
        assert result == expected_result

        result_graph = captured["compiled"]

        post_g1 = "eval_post_execution_guardrail1"
        log_post_g1 = "log_post_execution_guardrail1"
        post_g2 = "eval_post_execution_guardrail2"
        block_post_g2 = "block_post_execution_guardrail2"

        expected_edges = {
            ("START", "terminate"),
            ("terminate", post_g1),
            (log_post_g1, post_g2),
            (block_post_g2, "END"),
        }
        assert expected_edges.issubset(set(result_graph.edges))

        node_names = {name for name, _ in result_graph.nodes}
        for name in [
            "terminate",
            post_g1,
            post_g2,
            log_post_g1,
            block_post_g2,
        ]:
            assert name in node_names
        assert "eval_post_execution_llm_guardrail" not in node_names
        assert "noop_post_execution_llm_guardrail" not in node_names
