"""Integration tests for the EoG agent graph builder."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from uipath_langchain.agent.eog.agent import create_eog_agent
from uipath_langchain.agent.eog.ontology_client import OntologyClient
from uipath_langchain.agent.eog.types import InvestigationConfig


def _mock_ontology_client() -> AsyncMock:
    """Create a mock ontology client with a small 3-entity graph."""
    client = AsyncMock(spec=OntologyClient)
    client.discover = AsyncMock(
        return_value={
            "name": "test-ontology",
            "entities": ["e1", "e2", "e3"],
            "edges": [
                {"source": "e1", "target": "e2", "relationship": "causes"},
                {"source": "e2", "target": "e3", "relationship": "affects"},
            ],
        }
    )
    client.list_functions = AsyncMock(
        return_value=[
            {"name": "get_info", "description": "Get entity info"},
        ]
    )
    client.invoke_function = AsyncMock(
        return_value={"rows": [{"data": "test_data"}]}
    )
    return client


def _mock_llm(labels: dict[str, str] | None = None) -> AsyncMock:
    """Create a mock LLM that returns deterministic labels.

    Args:
        labels: Mapping from entity_id to label. Defaults to assigning
            "Source" to e1 and "DerivedEffect" to others.
    """
    default_labels = labels or {
        "e1": "Source",
        "e2": "DerivedEffect",
        "e3": "DerivedEffect",
    }

    async def ainvoke(messages: Any, **kwargs: Any) -> AsyncMock:
        # Extract entity_id from the prompt
        content = messages[-1].content if messages else ""
        entity_id = ""
        for eid in default_labels:
            if eid in content:
                entity_id = eid
                break

        label = default_labels.get(entity_id, "Defer")
        # Propagate to neighbours only from e1
        propagations = []
        if entity_id == "e1":
            propagations = [
                {"entity": "e2", "reason": "caused by e1"},
            ]
        elif entity_id == "e2":
            propagations = [
                {"entity": "e3", "reason": "affected by e2"},
            ]

        response = AsyncMock()
        response.content = json.dumps({
            "label": label,
            "evidence": f"Evidence for {entity_id}",
            "propagations": propagations,
        })
        return response

    model = AsyncMock()
    model.ainvoke = ainvoke
    return model


class TestCreateEoGAgent:
    def test_returns_state_graph(self) -> None:
        client = _mock_ontology_client()
        model = _mock_llm()
        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_entities=["e1"],
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        assert graph is not None

    def test_graph_compiles(self) -> None:
        client = _mock_ontology_client()
        model = _mock_llm()
        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_entities=["e1"],
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_full_traversal(self) -> None:
        """Integration: compile and run the graph with mock ontology."""
        client = _mock_ontology_client()
        model = _mock_llm()
        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_entities=["e1"],
            max_steps=20,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        # BFS should have terminated
        assert result["steps_taken"] >= 1

        # Ledger should have entries
        assert len(result["ledger"]) >= 1

        # Frontier should contain non-Defer entities
        assert len(result["frontier"]) >= 1
        frontier_entities = {f["entity"] for f in result["frontier"]}
        assert "e1" in frontier_entities

    @pytest.mark.asyncio
    async def test_propagation_causes_revisits(self) -> None:
        """Verify that propagation re-activates neighbours."""
        client = _mock_ontology_client()

        # LLM that flips labels to force re-visits
        call_count: dict[str, int] = {}

        async def flipping_ainvoke(messages: Any, **kwargs: Any) -> AsyncMock:
            content = messages[-1].content if messages else ""
            entity_id = ""
            for eid in ["e1", "e2", "e3"]:
                if eid in content:
                    entity_id = eid
                    break

            call_count[entity_id] = call_count.get(entity_id, 0) + 1
            count = call_count[entity_id]

            # Alternate labels to trigger flips
            label = "Source" if count % 2 == 1 else "DerivedEffect"
            propagations = (
                [{"entity": "e2", "reason": "update"}]
                if entity_id == "e1" and count == 1
                else []
            )

            response = AsyncMock()
            response.content = json.dumps({
                "label": label,
                "evidence": f"visit {count}",
                "propagations": propagations,
            })
            return response

        model = AsyncMock()
        model.ainvoke = flipping_ainvoke

        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_entities=["e1"],
            max_steps=10,
            max_flips=3,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        # e2 should have been visited because e1 propagated to it
        e2_entries = [
            e for e in result["ledger"] if e.entity_id == "e2"
        ]
        assert len(e2_entries) >= 1

    @pytest.mark.asyncio
    async def test_budget_cap_terminates(self) -> None:
        """Verify the graph stops at max_steps."""
        client = _mock_ontology_client()

        # LLM that always propagates, creating infinite loop potential
        async def always_propagate(
            messages: Any, **kwargs: Any
        ) -> AsyncMock:
            response = AsyncMock()
            response.content = json.dumps({
                "label": "Source",
                "evidence": "always",
                "propagations": [
                    {"entity": "e1", "reason": "loop"},
                    {"entity": "e2", "reason": "loop"},
                ],
            })
            return response

        model = AsyncMock()
        model.ainvoke = always_propagate

        cfg = InvestigationConfig(
            label_vocabulary=["Source", "Defer"],
            seed_entities=["e1"],
            max_steps=3,
            max_flips=10,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        assert result["steps_taken"] <= 3
