"""Tests for EoG node functions."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from uipath_langchain.agent.eog.nodes import (
    _make_bootstrap,
    _make_fetch_context,
    _make_policy,
    frontier_node,
    pop_node,
    propagate_node,
    should_continue,
    update_node,
)
from uipath_langchain.agent.eog.types import (
    Belief,
    EoGState,
    InvestigationConfig,
)


def _mock_ontology_client(
    discover_result: dict[str, Any] | None = None,
    functions_result: list[dict[str, Any]] | None = None,
    invoke_result: dict[str, Any] | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.discover = AsyncMock(
        return_value=discover_result or {"name": "onto", "edges": []}
    )
    client.list_functions = AsyncMock(
        return_value=functions_result or []
    )
    client.invoke_function = AsyncMock(
        return_value=invoke_result or {"rows": []}
    )
    return client


def _base_config() -> InvestigationConfig:
    return InvestigationConfig(
        label_vocabulary=["Source", "DerivedEffect", "Defer"],
        seed_entities=["e1", "e2"],
        max_steps=10,
    )


class TestBootstrapNode:
    @pytest.mark.asyncio
    async def test_bootstrap_initialises_beliefs(self) -> None:
        cfg = _base_config()
        client = _mock_ontology_client(
            discover_result={"name": "test"},
            functions_result=[{"name": "fn1", "description": "desc"}],
        )
        node = _make_bootstrap(client, "test-onto", cfg)
        state = EoGState()
        result = await node(state)

        assert "e1" in result["beliefs"]
        assert "e2" in result["beliefs"]
        assert result["beliefs"]["e1"].label == "Defer"
        assert result["active_set"] == ["e1", "e2"]
        assert result["steps_taken"] == 0
        assert "metadata" in result["ontology_graph"]
        assert "functions" in result["ontology_graph"]

    @pytest.mark.asyncio
    async def test_bootstrap_uses_state_config_as_fallback(self) -> None:
        client = _mock_ontology_client()
        node = _make_bootstrap(client, "onto", None)
        cfg = InvestigationConfig(
            label_vocabulary=["A"],
            seed_entities=["x"],
        )
        state = EoGState(investigation_config=cfg)
        result = await node(state)
        assert "x" in result["beliefs"]


class TestPopNode:
    @pytest.mark.asyncio
    async def test_pop_dequeues_first(self) -> None:
        state = EoGState(active_set=["a", "b", "c"])
        result = await pop_node(state)
        assert result["current_entity"] == "a"
        assert result["active_set"] == ["b", "c"]

    @pytest.mark.asyncio
    async def test_pop_empty(self) -> None:
        state = EoGState(active_set=[])
        result = await pop_node(state)
        assert result["current_entity"] == ""
        assert result["active_set"] == []


class TestShouldContinue:
    def test_continue_when_entity_and_budget(self) -> None:
        cfg = _base_config()
        state = EoGState(
            current_entity="e1",
            steps_taken=0,
            investigation_config=cfg,
        )
        assert should_continue(state) == "fetch_context"

    def test_frontier_when_no_current_entity(self) -> None:
        cfg = _base_config()
        state = EoGState(
            current_entity="",
            steps_taken=0,
            investigation_config=cfg,
        )
        assert should_continue(state) == "frontier"

    def test_frontier_when_budget_exhausted(self) -> None:
        cfg = _base_config()
        state = EoGState(
            current_entity="e1",
            steps_taken=10,
            investigation_config=cfg,
        )
        assert should_continue(state) == "frontier"

    def test_default_max_steps_when_no_config(self) -> None:
        state = EoGState(current_entity="e1", steps_taken=0)
        assert should_continue(state) == "fetch_context"

        state2 = EoGState(current_entity="e1", steps_taken=50)
        assert should_continue(state2) == "frontier"


class TestFetchContextNode:
    @pytest.mark.asyncio
    async def test_builds_context_packet(self) -> None:
        client = _mock_ontology_client(
            invoke_result={"rows": [{"val": 1}]}
        )
        cfg = _base_config()
        node = _make_fetch_context(client, "onto")
        state = EoGState(
            current_entity="e1",
            beliefs={"e1": Belief(label="Defer")},
            ontology_graph={
                "metadata": {"edges": []},
                "functions": [{"name": "fn_e1", "description": "e1 info"}],
            },
            investigation_config=cfg,
        )
        result = await node(state)
        ctx = result["context_packet"]
        assert ctx["entity_id"] == "e1"
        assert ctx["current_belief"] is not None
        assert len(ctx["function_results"]) == 1


class TestPolicyNode:
    @pytest.mark.asyncio
    async def test_policy_parses_llm_output(self) -> None:
        llm_response_content = json.dumps({
            "label": "Source",
            "evidence": "strong signal",
            "propagations": [{"entity": "e2", "reason": "linked"}],
        })
        model = AsyncMock()
        response_msg = AsyncMock()
        response_msg.content = llm_response_content
        model.ainvoke = AsyncMock(return_value=response_msg)

        node = _make_policy(model)
        cfg = _base_config()
        state = EoGState(
            current_entity="e1",
            context_packet={"entity_id": "e1", "data": "some data"},
            investigation_config=cfg,
        )
        result = await node(state)
        assert result["policy_result"]["label"] == "Source"
        assert len(result["policy_result"]["propagations"]) == 1

    @pytest.mark.asyncio
    async def test_policy_handles_parse_failure(self) -> None:
        model = AsyncMock()
        response_msg = AsyncMock()
        response_msg.content = "not valid json at all"
        model.ainvoke = AsyncMock(return_value=response_msg)

        node = _make_policy(model)
        state = EoGState(
            current_entity="e1",
            context_packet={"entity_id": "e1"},
            investigation_config=_base_config(),
        )
        result = await node(state)
        assert result["policy_result"]["label"] == "Defer"
        assert "could not be parsed" in result["policy_result"]["evidence"]


class TestUpdateNode:
    @pytest.mark.asyncio
    async def test_creates_ledger_entry(self) -> None:
        state = EoGState(
            current_entity="e1",
            beliefs={"e1": Belief(label="Defer")},
            policy_result={"label": "Source", "evidence": "found it"},
            steps_taken=0,
        )
        result = await update_node(state)
        assert result["beliefs"]["e1"].label == "Source"
        assert result["beliefs"]["e1"].flip_count == 1
        assert len(result["ledger"]) == 1
        assert result["ledger"][0].old_label == "Defer"
        assert result["ledger"][0].new_label == "Source"
        assert result["steps_taken"] == 1

    @pytest.mark.asyncio
    async def test_no_flip_when_same_label(self) -> None:
        state = EoGState(
            current_entity="e1",
            beliefs={"e1": Belief(label="Source", flip_count=0)},
            policy_result={"label": "Source", "evidence": "confirmed"},
            steps_taken=2,
        )
        result = await update_node(state)
        assert result["beliefs"]["e1"].flip_count == 0
        assert result["steps_taken"] == 3

    @pytest.mark.asyncio
    async def test_new_entity_no_flip(self) -> None:
        state = EoGState(
            current_entity="new_e",
            beliefs={},
            policy_result={"label": "Source", "evidence": "new"},
            steps_taken=0,
        )
        result = await update_node(state)
        assert result["beliefs"]["new_e"].flip_count == 0


class TestPropagateNode:
    @pytest.mark.asyncio
    async def test_propagates_to_neighbours(self) -> None:
        state = EoGState(
            current_entity="e1",
            beliefs={"e1": Belief(label="Source", evidence="test")},
            policy_result={
                "propagations": [
                    {"entity": "e2", "reason": "linked"},
                    {"entity": "e3", "reason": "related"},
                ],
            },
            active_set=[],
            investigation_config=_base_config(),
        )
        result = await propagate_node(state)
        assert "e2" in result["inbox"]
        assert "e3" in result["inbox"]
        assert "e2" in result["active_set"]
        assert "e3" in result["active_set"]
        assert len(result["explanatory_edges"]) == 2

    @pytest.mark.asyncio
    async def test_does_not_reactivate_over_max_flips(self) -> None:
        cfg = InvestigationConfig(
            label_vocabulary=["A"], max_flips=2
        )
        state = EoGState(
            current_entity="e1",
            beliefs={
                "e1": Belief(label="A"),
                "e2": Belief(label="A", flip_count=3),
            },
            policy_result={
                "propagations": [{"entity": "e2", "reason": "test"}],
            },
            active_set=[],
            investigation_config=cfg,
        )
        result = await propagate_node(state)
        # e2 should NOT be re-activated because flip_count >= max_flips
        assert "e2" not in result["active_set"]

    @pytest.mark.asyncio
    async def test_no_duplicates_in_active_set(self) -> None:
        state = EoGState(
            current_entity="e1",
            beliefs={"e1": Belief(label="A")},
            policy_result={
                "propagations": [{"entity": "e2", "reason": "x"}],
            },
            active_set=["e2"],
            investigation_config=_base_config(),
        )
        result = await propagate_node(state)
        # e2 already in active_set, should not be duplicated
        assert result["active_set"].count("e2") == 1


class TestFrontierNode:
    @pytest.mark.asyncio
    async def test_filters_defer_labels(self) -> None:
        state = EoGState(
            beliefs={
                "e1": Belief(label="Source", evidence="found"),
                "e2": Belief(label="Defer", evidence="unknown"),
                "e3": Belief(label="DerivedEffect", evidence="derived"),
            },
            investigation_config=_base_config(),
        )
        result = await frontier_node(state)
        entities = [f["entity"] for f in result["frontier"]]
        assert "e1" in entities
        assert "e3" in entities
        assert "e2" not in entities

    @pytest.mark.asyncio
    async def test_empty_frontier_when_all_defer(self) -> None:
        state = EoGState(
            beliefs={
                "e1": Belief(label="Defer"),
            },
            investigation_config=_base_config(),
        )
        result = await frontier_node(state)
        assert result["frontier"] == []
