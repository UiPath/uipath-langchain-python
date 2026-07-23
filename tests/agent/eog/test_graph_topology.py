"""Tests for OWL Functional Notation parser and graph topology."""

from __future__ import annotations

from uipath_langchain.agent.eog.graph_topology import (
    OntologyGraph,
    parse_ofn,
)

_SAMPLE_OFN = """\
Prefix(:=<https://ontology.uipath.com/ont#>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)

Ontology(<https://ontology.uipath.com/ont>

Declaration(Class(:Supplier))
Declaration(Class(:Invoice))
Declaration(Class(:Commodity))
Declaration(ObjectProperty(:invoice_paidTo))
Declaration(ObjectProperty(:spend_forCommodity))
Declaration(DataProperty(:Supplier.supplierId))
Declaration(DataProperty(:Supplier.name))
Declaration(DataProperty(:Invoice.invoiceId))
Declaration(DataProperty(:Invoice.amount))
Declaration(DataProperty(:Commodity.commodityId))

AnnotationAssertion(rdfs:label :invoice_paidTo "paid to")
ObjectPropertyDomain(:invoice_paidTo :Invoice)
ObjectPropertyRange(:invoice_paidTo :Supplier)

AnnotationAssertion(rdfs:label :spend_forCommodity "for commodity")
ObjectPropertyDomain(:spend_forCommodity :Invoice)
ObjectPropertyRange(:spend_forCommodity :Commodity)

DataPropertyDomain(:Supplier.supplierId :Supplier)
DataPropertyRange(:Supplier.supplierId xsd:string)
DataPropertyDomain(:Supplier.name :Supplier)
DataPropertyRange(:Supplier.name xsd:string)
DataPropertyDomain(:Invoice.invoiceId :Invoice)
DataPropertyRange(:Invoice.invoiceId xsd:string)
DataPropertyDomain(:Invoice.amount :Invoice)
DataPropertyRange(:Invoice.amount xsd:decimal)
DataPropertyDomain(:Commodity.commodityId :Commodity)
DataPropertyRange(:Commodity.commodityId xsd:string)

AnnotationAssertion(rdfs:label :Supplier "Supplier")
AnnotationAssertion(rdfs:label :Invoice "Invoice")
AnnotationAssertion(rdfs:label :Commodity "Commodity")

)
"""


class TestParseOfn:
    def test_extracts_classes(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        assert set(graph.nodes.keys()) == {"Supplier", "Invoice", "Commodity"}

    def test_extracts_labels(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        assert graph.nodes["Supplier"].label == "Supplier"
        assert graph.nodes["Invoice"].label == "Invoice"

    def test_extracts_data_properties(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        sup_props = {p.name for p in graph.nodes["Supplier"].data_properties}
        assert sup_props == {"supplierId", "name"}
        inv_props = {p.name for p in graph.nodes["Invoice"].data_properties}
        assert inv_props == {"invoiceId", "amount"}

    def test_extracts_data_property_ranges(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        amount_prop = next(
            p for p in graph.nodes["Invoice"].data_properties
            if p.name == "amount"
        )
        assert amount_prop.range == "xsd:decimal"

    def test_extracts_edges(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        assert len(graph.edges) == 2
        edge_names = {e.name for e in graph.edges}
        assert edge_names == {"invoice_paidTo", "spend_forCommodity"}

    def test_edge_domain_range(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        paid_to = next(e for e in graph.edges if e.name == "invoice_paidTo")
        assert paid_to.source == "Invoice"
        assert paid_to.target == "Supplier"
        assert paid_to.label == "paid to"

    def test_edge_labels(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        for_com = next(e for e in graph.edges if e.name == "spend_forCommodity")
        assert for_com.label == "for commodity"


class TestOntologyGraphAdjacency:
    def test_neighbors(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        assert set(graph.neighbors("Invoice")) == {"Supplier", "Commodity"}
        assert set(graph.neighbors("Supplier")) == {"Invoice"}
        assert set(graph.neighbors("Commodity")) == {"Invoice"}

    def test_outgoing(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        out = graph.outgoing("Invoice")
        assert len(out) == 2
        targets = {e.target for e in out}
        assert targets == {"Supplier", "Commodity"}

    def test_incoming(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        inc = graph.incoming("Supplier")
        assert len(inc) == 1
        assert inc[0].source == "Invoice"

    def test_edges_of(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        all_edges = graph.edges_of("Invoice")
        assert len(all_edges) == 2  # 2 outgoing, 0 incoming


class TestFunctionMapping:
    def test_functions_for_entity(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        graph.functions = [
            {
                "name": "invoiceDetail",
                "label": "Invoice detail",
                "statement": "SELECT ?x WHERE { ?inv a ont:Invoice }",
                "params": [{"name": "invoiceId", "type": "xsd:string"}],
            },
            {
                "name": "supplierProfile",
                "label": "Supplier profile",
                "statement": "SELECT ?x WHERE { ?s a ont:Supplier }",
                "params": [{"name": "supplierId", "type": "xsd:string"}],
            },
        ]
        graph._build_adjacency()

        inv_fns = graph.functions_for("Invoice")
        assert len(inv_fns) == 1
        assert inv_fns[0]["name"] == "invoiceDetail"

        sup_fns = graph.functions_for("Supplier")
        assert len(sup_fns) == 1
        assert sup_fns[0]["name"] == "supplierProfile"


class TestEntityIdResolution:
    def test_prefix_resolution(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        assert graph.entity_for_id("INV-2004") == "Invoice"
        assert graph.entity_for_id("SUP-001") == "Supplier"
        assert graph.entity_for_id("COM-110") == "Commodity"

    def test_unknown_prefix(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        assert graph.entity_for_id("UNKNOWN-1") is None


class TestSerialization:
    def test_round_trip(self) -> None:
        graph = parse_ofn(_SAMPLE_OFN)
        graph.functions = [{"name": "test_fn", "label": "Test"}]
        graph._build_adjacency()

        data = graph.to_dict()
        restored = OntologyGraph.from_dict(data)

        assert set(restored.nodes.keys()) == set(graph.nodes.keys())
        assert len(restored.edges) == len(graph.edges)
        assert restored.neighbors("Invoice") == graph.neighbors("Invoice")
