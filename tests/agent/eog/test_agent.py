"""Integration tests for the EoG agent graph builder."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from uipath_langchain.agent.eog.agent import create_eog_agent
from uipath_langchain.agent.eog.ontology_client import OntologyClient
from uipath_langchain.agent.eog.types import InvestigationConfig


# ── Test fixtures ─────────────────────────────────────────────────

_INVOICE_FUNCTIONS = [
    {
        "name": "invoiceDetail",
        "label": "Invoice detail",
        "description": "Retrieve invoice with supplier and PO",
        "params": [{"name": "invoiceId", "type": "xsd:string", "required": True}],
        "outputs": [
            {"name": "supplierId", "type": "xsd:string"},
            {"name": "poId", "type": "xsd:string"},
            {"name": "amount", "type": "xsd:decimal"},
        ],
        "touches": ["Invoice", "Supplier", "PurchaseOrder"],
    },
]

_SUPPLIER_FUNCTIONS = [
    {
        "name": "supplierProfile",
        "label": "Supplier profile",
        "description": "Supplier details",
        "params": [{"name": "supplierId", "type": "xsd:string", "required": True}],
        "outputs": [{"name": "name", "type": "xsd:string"}],
        "touches": ["Supplier"],
    },
]


def _mock_ontology_client() -> AsyncMock:
    """Create a mock ontology client with lazy function discovery."""
    client = AsyncMock(spec=OntologyClient)

    async def _list_functions(
        ontology: str, *, touches: str | None = None,
    ) -> list[dict[str, Any]]:
        if touches == "Invoice":
            return _INVOICE_FUNCTIONS
        if touches == "Supplier":
            return _SUPPLIER_FUNCTIONS
        return _INVOICE_FUNCTIONS + _SUPPLIER_FUNCTIONS

    client.list_functions = AsyncMock(side_effect=_list_functions)
    client.invoke_function = AsyncMock(
        return_value={
            "rows": [{"supplierId": "SUP-001", "poId": "PO-001", "amount": 16224.0}],
        }
    )
    return client


def _mock_llm(labels: dict[str, str] | None = None) -> AsyncMock:
    """Create a mock LLM that returns deterministic labels."""
    default_labels = labels or {
        "INV-001": "Source",
        "SUP-001": "DerivedEffect",
        "PO-001": "DerivedEffect",
    }

    async def ainvoke(messages: Any, **kwargs: Any) -> AsyncMock:
        content = messages[-1].content if messages else ""
        entity_id = ""
        for eid in default_labels:
            if eid in content:
                entity_id = eid
                break

        label = default_labels.get(entity_id, "Defer")
        response = AsyncMock()
        response.content = json.dumps({
            "label": label,
            "evidence": f"Evidence for {entity_id}",
        })
        return response

    model = AsyncMock()
    model.ainvoke = ainvoke
    return model


# ── Tests ─────────────────────────────────────────────────────────

class TestCreateEoGAgent:
    def test_returns_state_graph(self) -> None:
        client = _mock_ontology_client()
        model = _mock_llm()
        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_records=["INV-001"],
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
            seed_records=["INV-001"],
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
            seed_records=["INV-001"],
            max_steps=20,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        assert result["steps_taken"] >= 1
        assert len(result["ledger"]) >= 1
        # INV-001 should be labeled Source
        assert result["beliefs"]["INV-001"].label == "Source"

    @pytest.mark.asyncio
    async def test_lazy_discovery(self) -> None:
        """Verify functions are fetched per entity type, not upfront."""
        client = _mock_ontology_client()
        model = _mock_llm()
        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_records=["INV-001"],
            max_steps=5,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        # Should have cached functions for Invoice (the seed type)
        assert "Invoice" in result["function_cache"]

    @pytest.mark.asyncio
    async def test_entity_discovery_from_results(self) -> None:
        """Verify new entities are discovered from function results."""
        client = _mock_ontology_client()
        model = _mock_llm()
        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_records=["INV-001"],
            max_steps=10,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        # invoiceDetail returns supplierId=SUP-001 and poId=PO-001
        # These should be discovered as new entities
        assert "SUP-001" in result["discovered_records"]
        assert "PO-001" in result["discovered_records"]

    @pytest.mark.asyncio
    async def test_budget_cap_terminates(self) -> None:
        """Verify the graph stops at max_steps."""
        client = _mock_ontology_client()

        async def always_source(messages: Any, **kwargs: Any) -> AsyncMock:
            response = AsyncMock()
            response.content = json.dumps({
                "label": "Source",
                "evidence": "always",
            })
            return response

        model = AsyncMock()
        model.ainvoke = always_source

        cfg = InvestigationConfig(
            label_vocabulary=["Source", "Defer"],
            seed_records=["INV-001"],
            max_steps=3,
            max_flips=10,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        assert result["steps_taken"] <= 3
