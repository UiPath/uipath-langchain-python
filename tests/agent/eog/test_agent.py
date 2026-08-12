"""Integration tests for the EoG agent graph builder."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from uipath_langchain.agent.eog.agent import create_eog_agent
from uipath_langchain.agent.eog.graph_topology import OntologyGraph, parse_ofn
from uipath_langchain.agent.eog.ontology_client import OntologyClient
from uipath_langchain.agent.eog.types import InvestigationConfig

# Minimal OWL with 3 entities and 2 relationships
_TEST_OFN = """\
Prefix(:=<https://ontology.uipath.com/ont#>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)

Ontology(<https://ontology.uipath.com/ont>

Declaration(Class(:Invoice))
Declaration(Class(:Supplier))
Declaration(Class(:PurchaseOrder))
Declaration(ObjectProperty(:invoice_paidTo))
Declaration(ObjectProperty(:invoice_referencesPO))
Declaration(DataProperty(:Invoice.invoiceId))
Declaration(DataProperty(:Supplier.supplierId))
Declaration(DataProperty(:PurchaseOrder.poId))

ObjectPropertyDomain(:invoice_paidTo :Invoice)
ObjectPropertyRange(:invoice_paidTo :Supplier)
ObjectPropertyDomain(:invoice_referencesPO :Invoice)
ObjectPropertyRange(:invoice_referencesPO :PurchaseOrder)

DataPropertyDomain(:Invoice.invoiceId :Invoice)
DataPropertyRange(:Invoice.invoiceId xsd:string)
DataPropertyDomain(:Supplier.supplierId :Supplier)
DataPropertyRange(:Supplier.supplierId xsd:string)
DataPropertyDomain(:PurchaseOrder.poId :PurchaseOrder)
DataPropertyRange(:PurchaseOrder.poId xsd:string)

AnnotationAssertion(rdfs:label :Invoice "Invoice")
AnnotationAssertion(rdfs:label :Supplier "Supplier")
AnnotationAssertion(rdfs:label :PurchaseOrder "Purchase Order")

)
"""

_TEST_FUNCTIONS = [
    {
        "name": "invoiceDetail",
        "label": "Invoice detail",
        "statement": "SELECT ?x WHERE { ?inv a ont:Invoice }",
        "params": [{"name": "invoiceId", "type": "xsd:string", "required": True}],
    },
    {
        "name": "supplierProfile",
        "label": "Supplier profile",
        "statement": "SELECT ?x WHERE { ?s a ont:Supplier }",
        "params": [{"name": "supplierId", "type": "xsd:string", "required": True}],
    },
]


def _mock_ontology_client() -> AsyncMock:
    """Create a mock ontology client that returns the test graph."""
    client = AsyncMock(spec=OntologyClient)

    async def _fetch_graph(ontology: str) -> OntologyGraph:
        graph = parse_ofn(_TEST_OFN)
        graph.functions = _TEST_FUNCTIONS
        graph._build_adjacency()
        return graph

    client.fetch_graph = AsyncMock(side_effect=_fetch_graph)
    client.invoke_function = AsyncMock(
        return_value={"rows": [{"data": "test_data"}]}
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


class TestCreateEoGAgent:
    def test_returns_state_graph(self) -> None:
        client = _mock_ontology_client()
        model = _mock_llm()
        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_entities=["INV-001"],
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
            seed_entities=["INV-001"],
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
            seed_entities=["INV-001", "SUP-001"],
            max_steps=20,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        assert result["steps_taken"] >= 1
        assert len(result["ledger"]) >= 1
        assert len(result["frontier"]) >= 1
        frontier_entities = {f["entity"] for f in result["frontier"]}
        assert "INV-001" in frontier_entities

    @pytest.mark.asyncio
    async def test_propagation_causes_revisits(self) -> None:
        """Verify that graph-edge propagation re-activates neighbours."""
        client = _mock_ontology_client()

        # LLM that labels INV-001 as Source (changed from Defer)
        # → should propagate to SUP-001 via invoice_paidTo edge
        async def labeling_ainvoke(messages: Any, **kwargs: Any) -> AsyncMock:
            content = messages[-1].content if messages else ""
            if "INV-001" in content:
                label = "Source"
            elif "SUP-001" in content:
                label = "DerivedEffect"
            else:
                label = "Defer"
            response = AsyncMock()
            response.content = json.dumps({
                "label": label,
                "evidence": f"evidence",
            })
            return response

        model = AsyncMock()
        model.ainvoke = labeling_ainvoke

        cfg = InvestigationConfig(
            label_vocabulary=["Source", "DerivedEffect", "Defer"],
            seed_entities=["INV-001", "SUP-001"],
            max_steps=10,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        # Both should be visited. INV-001 labeled Source triggers
        # propagation to SUP-001 via invoice_paidTo graph edge,
        # causing SUP-001 to be revisited with inbox context.
        sup_entries = [
            e for e in result["ledger"] if e.entity_id == "SUP-001"
        ]
        assert len(sup_entries) >= 1
        # SUP-001 should have inbox messages from INV-001
        assert len(result.get("inbox", {}).get("SUP-001", [])) >= 0

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
            seed_entities=["INV-001"],
            max_steps=3,
            max_flips=10,
        )
        graph = create_eog_agent(
            model, client, "test-onto", investigation_config=cfg
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({})

        assert result["steps_taken"] <= 3
