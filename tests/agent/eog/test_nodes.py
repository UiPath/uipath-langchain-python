"""Tests for EoG node functions."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from uipath_langchain.agent.eog.graph_topology import OntologyGraph, parse_ofn
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
    ExplanatoryEdge,
    InvestigationConfig,
)

# ── Minimal OWL for testing ────────────────────────────────────────

_TEST_OFN = """\
Prefix(:=<https://ontology.uipath.com/ont#>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)

Ontology(<https://ontology.uipath.com/ont>

Declaration(Class(:Supplier))
Declaration(Class(:Invoice))
Declaration(Class(:PurchaseOrder))
Declaration(ObjectProperty(:invoice_paidTo))
Declaration(ObjectProperty(:invoice_referencesPO))
Declaration(DataProperty(:Supplier.supplierId))
Declaration(DataProperty(:Supplier.name))
Declaration(DataProperty(:Invoice.invoiceId))
Declaration(DataProperty(:Invoice.amount))
Declaration(DataProperty(:PurchaseOrder.poId))

AnnotationAssertion(rdfs:label :invoice_paidTo "paid to")
ObjectPropertyDomain(:invoice_paidTo :Invoice)
ObjectPropertyRange(:invoice_paidTo :Supplier)

AnnotationAssertion(rdfs:label :invoice_referencesPO "references PO")
ObjectPropertyDomain(:invoice_referencesPO :Invoice)
ObjectPropertyRange(:invoice_referencesPO :PurchaseOrder)

DataPropertyDomain(:Supplier.supplierId :Supplier)
DataPropertyRange(:Supplier.supplierId xsd:string)
DataPropertyDomain(:Supplier.name :Supplier)
DataPropertyRange(:Supplier.name xsd:string)
DataPropertyDomain(:Invoice.invoiceId :Invoice)
DataPropertyRange(:Invoice.invoiceId xsd:string)
DataPropertyDomain(:Invoice.amount :Invoice)
DataPropertyRange(:Invoice.amount xsd:decimal)
DataPropertyDomain(:PurchaseOrder.poId :PurchaseOrder)
DataPropertyRange(:PurchaseOrder.poId xsd:string)

AnnotationAssertion(rdfs:label :Supplier "Supplier")
AnnotationAssertion(rdfs:label :Invoice "Invoice")
AnnotationAssertion(rdfs:label :PurchaseOrder "Purchase Order")

)
"""

_TEST_FUNCTIONS = [
    {
        "name": "invoiceDetail",
        "label": "Invoice detail",
        "language": "SPARQL",
        "statement": "PREFIX ont: <https://ontology.uipath.com/ont#> SELECT ?amount WHERE { ?inv a ont:Invoice }",
        "params": [{"name": "invoiceId", "type": "xsd:string", "required": True}],
    },
    {
        "name": "supplierProfile",
        "label": "Supplier profile",
        "language": "SPARQL",
        "statement": "PREFIX ont: <https://ontology.uipath.com/ont#> SELECT ?name WHERE { ?s a ont:Supplier }",
        "params": [{"name": "supplierId", "type": "xsd:string", "required": True}],
    },
]


def _mock_ontology_client(
    invoke_result: dict[str, Any] | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.fetch_graph = AsyncMock(side_effect=_make_test_graph)
    client.fetch_artifact = AsyncMock(return_value=_TEST_OFN)
    client.list_functions = AsyncMock(return_value=_TEST_FUNCTIONS)
    client.invoke_function = AsyncMock(
        return_value=invoke_result or {"rows": [{"amount": 100}]}
    )
    return client


async def _make_test_graph(ontology: str) -> OntologyGraph:
    graph = parse_ofn(_TEST_OFN)
    graph.functions = _TEST_FUNCTIONS
    graph._build_adjacency()
    return graph


def _base_config() -> InvestigationConfig:
    return InvestigationConfig(
        label_vocabulary=["Source", "DerivedEffect", "Defer"],
        seed_entities=["INV-2004", "SUP-001"],
        max_steps=10,
        max_flips=3,
    )


def _test_graph_dict() -> dict[str, Any]:
    graph = parse_ofn(_TEST_OFN)
    graph.functions = _TEST_FUNCTIONS
    graph._build_adjacency()
    return graph.to_dict()


# ── Bootstrap tests ────────────────────────────────────────────────


class TestBootstrapNode:
    async def test_bootstrap_initialises_beliefs(self) -> None:
        client = _mock_ontology_client()
        cfg = _base_config()
        bootstrap = _make_bootstrap(client, "test", cfg)
        state = EoGState()

        result = await bootstrap(state)

        assert "INV-2004" in result["beliefs"]
        assert "SUP-001" in result["beliefs"]
        assert result["beliefs"]["INV-2004"].label == "Defer"
        assert result["active_set"] == ["INV-2004", "SUP-001"]
        assert result["steps_taken"] == 0
        # Graph topology should be populated
        assert "nodes" in result["ontology_graph"]
        assert "edges" in result["ontology_graph"]
        assert "Invoice" in result["ontology_graph"]["nodes"]

    async def test_bootstrap_fetches_graph_topology(self) -> None:
        client = _mock_ontology_client()
        cfg = _base_config()
        bootstrap = _make_bootstrap(client, "test", cfg)
        state = EoGState()

        result = await bootstrap(state)
        graph_data = result["ontology_graph"]

        assert len(graph_data["nodes"]) == 3  # Supplier, Invoice, PurchaseOrder
        assert len(graph_data["edges"]) == 2  # invoice_paidTo, invoice_referencesPO
        assert len(graph_data["functions"]) == 2


# ── Pop tests ──────────────────────────────────────────────────────


class TestPopNode:
    async def test_pops_first_entity(self) -> None:
        state = EoGState(active_set=["a", "b", "c"])
        result = await pop_node(state)
        assert result["current_entity"] == "a"
        assert result["active_set"] == ["b", "c"]

    async def test_empty_set_yields_empty_string(self) -> None:
        state = EoGState(active_set=[])
        result = await pop_node(state)
        assert result["current_entity"] == ""


# ── should_continue tests ──────────────────────────────────────────


class TestShouldContinue:
    def test_continues_when_entity_and_budget(self) -> None:
        state = EoGState(
            current_entity="x",
            steps_taken=0,
            investigation_config=_base_config(),
        )
        assert should_continue(state) == "fetch_context"

    def test_stops_at_budget(self) -> None:
        state = EoGState(
            current_entity="x",
            steps_taken=10,
            investigation_config=_base_config(),
        )
        assert should_continue(state) == "frontier"

    def test_stops_when_empty(self) -> None:
        state = EoGState(
            current_entity="",
            steps_taken=0,
            investigation_config=_base_config(),
        )
        assert should_continue(state) == "frontier"


# ── Fetch context tests ───────────────────────────────────────────


class TestFetchContextNode:
    async def test_builds_context_packet(self) -> None:
        client = _mock_ontology_client()
        fetch = _make_fetch_context(client, "test")

        state = EoGState(
            current_entity="INV-2004",
            ontology_graph=_test_graph_dict(),
            beliefs={
                "INV-2004": Belief(label="Defer", evidence="seed"),
                "SUP-001": Belief(label="Source", evidence="known"),
            },
            investigation_config=_base_config(),
        )

        result = await fetch(state)
        ctx = result["context_packet"]

        assert ctx["entity_id"] == "INV-2004"
        assert ctx["entity_type"] == "Invoice"
        # Should have neighbor beliefs (Supplier is 1-hop via invoice_paidTo)
        assert "SUP-001" in ctx["neighbor_beliefs"]
        assert ctx["neighbor_beliefs"]["SUP-001"]["label"] == "Source"
        # Should have invoked invoiceDetail (touches Invoice)
        assert len(ctx["function_results"]) > 0

    async def test_binds_params_correctly(self) -> None:
        client = _mock_ontology_client()
        fetch = _make_fetch_context(client, "test")

        state = EoGState(
            current_entity="SUP-001",
            ontology_graph=_test_graph_dict(),
            beliefs={"SUP-001": Belief(label="Defer", evidence="seed")},
            investigation_config=_base_config(),
        )

        result = await fetch(state)
        # Should have called supplierProfile with supplierId=SUP-001
        call_args = client.invoke_function.call_args_list
        supplier_calls = [
            c for c in call_args
            if c[0][1] == "supplierProfile"
        ]
        assert len(supplier_calls) == 1
        assert supplier_calls[0][0][2] == {"supplierId": "SUP-001"}


# ── Policy tests ───────────────────────────────────────────────────


class TestPolicyNode:
    async def test_parses_llm_json(self) -> None:
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AsyncMock(
            content='{"label": "Source", "evidence": "found the root cause"}'
        ))
        policy = _make_policy(llm)
        state = EoGState(
            current_entity="INV-2004",
            context_packet={"entity_id": "INV-2004", "entity_type": "Invoice"},
            investigation_config=_base_config(),
        )

        result = await policy(state)
        assert result["policy_result"]["label"] == "Source"
        assert "root cause" in result["policy_result"]["evidence"]

    async def test_enforces_label_vocabulary(self) -> None:
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AsyncMock(
            content='{"label": "InvalidLabel", "evidence": "bad"}'
        ))
        policy = _make_policy(llm)
        state = EoGState(
            current_entity="x",
            context_packet={"entity_id": "x", "entity_type": "Unknown"},
            investigation_config=_base_config(),
        )

        result = await policy(state)
        assert result["policy_result"]["label"] == "Defer"  # Forced to Defer

    async def test_handles_parse_failure(self) -> None:
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AsyncMock(content="not json"))
        policy = _make_policy(llm)
        state = EoGState(
            current_entity="x",
            context_packet={"entity_id": "x", "entity_type": "Unknown"},
            investigation_config=_base_config(),
        )

        result = await policy(state)
        assert result["policy_result"]["label"] == "Defer"


# ── Update tests ───────────────────────────────────────────────────


class TestUpdateNode:
    async def test_creates_ledger_entry(self) -> None:
        state = EoGState(
            current_entity="INV-2004",
            policy_result={"label": "Source", "evidence": "found it"},
            beliefs={"INV-2004": Belief(label="Defer", evidence="seed")},
            investigation_config=_base_config(),
        )

        result = await update_node(state)
        assert len(result["ledger"]) == 1
        assert result["ledger"][0].old_label == "Defer"
        assert result["ledger"][0].new_label == "Source"

    async def test_tracks_flip_count(self) -> None:
        state = EoGState(
            current_entity="x",
            policy_result={"label": "DerivedEffect", "evidence": "changed"},
            beliefs={"x": Belief(label="Source", evidence="was source", flip_count=1)},
            investigation_config=_base_config(),
        )

        result = await update_node(state)
        assert result["beliefs"]["x"].flip_count == 2

    async def test_damping_forces_defer(self) -> None:
        state = EoGState(
            current_entity="x",
            policy_result={"label": "Source", "evidence": "oscillating"},
            beliefs={"x": Belief(label="DerivedEffect", evidence="prev", flip_count=3)},
            investigation_config=_base_config(),
        )

        result = await update_node(state)
        # flip_count exceeds max_flips(3) → damped to Defer
        assert result["beliefs"]["x"].label == "Defer"
        assert "Damped" in result["beliefs"]["x"].evidence


# ── Propagate tests ────────────────────────────────────────────────


class TestPropagateNode:
    async def test_propagates_to_graph_neighbors(self) -> None:
        """Propagation follows GRAPH EDGES, not LLM suggestions."""
        from uipath_langchain.agent.eog.types import LedgerEntry

        state = EoGState(
            current_entity="INV-2004",
            ontology_graph=_test_graph_dict(),
            beliefs={
                "INV-2004": Belief(label="Source", evidence="found it"),
                "SUP-001": Belief(label="Defer", evidence="seed"),
            },
            # Ledger shows label changed (Defer → Source)
            ledger=[LedgerEntry(
                timestamp=0, entity_id="INV-2004",
                old_label="Defer", new_label="Source", evidence="found it",
            )],
            investigation_config=_base_config(),
            policy_result={},
        )

        result = await propagate_node(state)

        # SUP-001 should be re-activated (Invoice→Supplier via invoice_paidTo)
        assert "SUP-001" in result.get("active_set", [])
        # SUP-001 should have an inbox message from INV-2004
        assert "SUP-001" in result.get("inbox", {})
        msg = result["inbox"]["SUP-001"][0]
        assert msg["from"] == "INV-2004"
        assert msg["label"] == "Source"

    async def test_no_propagation_when_label_unchanged(self) -> None:
        from uipath_langchain.agent.eog.types import LedgerEntry

        state = EoGState(
            current_entity="INV-2004",
            ontology_graph=_test_graph_dict(),
            beliefs={
                "INV-2004": Belief(label="Source", evidence="same"),
                "SUP-001": Belief(label="Defer", evidence="seed"),
            },
            # Ledger shows NO change (Source → Source)
            ledger=[LedgerEntry(
                timestamp=0, entity_id="INV-2004",
                old_label="Source", new_label="Source", evidence="same",
            )],
            investigation_config=_base_config(),
            policy_result={},
        )

        result = await propagate_node(state)
        # No propagation when label didn't change
        assert not result  # Empty dict

    async def test_does_not_reactivate_over_max_flips(self) -> None:
        from uipath_langchain.agent.eog.types import LedgerEntry

        state = EoGState(
            current_entity="INV-2004",
            ontology_graph=_test_graph_dict(),
            beliefs={
                "INV-2004": Belief(label="Source", evidence="found"),
                "SUP-001": Belief(label="Defer", evidence="seed", flip_count=4),
            },
            ledger=[LedgerEntry(
                timestamp=0, entity_id="INV-2004",
                old_label="Defer", new_label="Source", evidence="found",
            )],
            investigation_config=_base_config(),
            policy_result={},
        )

        result = await propagate_node(state)
        # SUP-001 has flip_count=4 > max_flips=3 → not re-activated
        active = result.get("active_set", [])
        assert "SUP-001" not in active


# ── Frontier tests ─────────────────────────────────────────────────


class TestFrontierNode:
    async def test_filters_non_defer(self) -> None:
        state = EoGState(
            beliefs={
                "a": Belief(label="Source", evidence="root cause"),
                "b": Belief(label="DerivedEffect", evidence="symptom"),
                "c": Belief(label="Defer", evidence="unknown"),
            },
            ontology_graph=_test_graph_dict(),
            investigation_config=_base_config(),
        )

        result = await frontier_node(state)
        labels = {f["entity"]: f["label"] for f in result["frontier"]}
        assert "a" in labels
        assert "b" in labels
        assert "c" not in labels  # Defer excluded

    async def test_removes_explained_entities(self) -> None:
        state = EoGState(
            beliefs={
                "a": Belief(label="Source", evidence="root cause"),
                "b": Belief(label="DerivedEffect", evidence="caused by a"),
            },
            explanatory_edges=[
                ExplanatoryEdge(source="a", target="b", relationship="causes"),
            ],
            ontology_graph=_test_graph_dict(),
            investigation_config=_base_config(),
        )

        result = await frontier_node(state)
        entities = {f["entity"] for f in result["frontier"]}
        # b is explained by a (Source→DerivedEffect) → removed from frontier
        assert "a" in entities
        assert "b" not in entities
