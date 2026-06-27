"""Tests for the OWL ontology compiler (refund hero case)."""

from __future__ import annotations

import pytest

from uipath_langchain.agent.tools.datafabric_tool.compiled_ontology import (
    CompiledOntology,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology_compiler import (
    OntologyCompileError,
    compile_ontology,
)

# A small refund-domain ontology in the .ttl dialect (subClassOf + actions +
# df:hasField + df:requiresHITL), mirroring p1-owl-write-extension.ttl.
REFUND_OWL = """
@prefix df:   <https://ontology.uipath.com/datafabric#> .
@prefix ex:   <https://ontology.example.com/refund#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# ---- Entities ----

# Federated / read-only: NOT a df:WritableEntity
ex:Customer a owl:Class ;
    rdfs:subClassOf df:ReadableEntity ;
    df:entityKey "Customer" .

# Writable entities
ex:RefundRequest a owl:Class ;
    rdfs:subClassOf df:WritableEntity ;
    df:entityKey "RefundRequest" .

ex:Order a owl:Class ;
    rdfs:subClassOf df:WritableEntity ;
    df:entityKey "Order" .

ex:CustomerRisk a owl:Class ;
    rdfs:subClassOf df:WritableEntity ;
    df:entityKey "CustomerRisk" .

# ---- Fields ----

ex:field_RefundRequest_Amount a df:MeasureField ;
    df:fieldKey "ApprovedAmount" ;
    df:measureSemantics "additive" .

ex:field_Order_Status a df:StateField ;
    df:fieldKey "Status" ;
    df:choiceSetKey "OrderStatusChoiceSet" .

ex:field_RefundRequest_OrderId a df:ReferenceField ;
    df:fieldKey "OrderId" ;
    df:referencesEntity ex:Order .

ex:field_CustomerRisk_Score a df:MeasureField ;
    df:fieldKey "RiskScore" ;
    df:measureSemantics "additive" .

# Field -> entity binding
ex:RefundRequest df:hasField ex:field_RefundRequest_Amount ,
                             ex:field_RefundRequest_OrderId .
ex:Order df:hasField ex:field_Order_Status .
ex:CustomerRisk df:hasField ex:field_CustomerRisk_Score .

# ---- Actions ----

ex:CreateRefund a df:InsertAction ;
    df:writeOperation "insert" ;
    df:targetEntity ex:RefundRequest ;
    df:requiresHITL false .

ex:UpdateOrder a df:UpdateAction ;
    df:writeOperation "update" ;
    df:targetEntity ex:Order ;
    df:requiresHITL false .

ex:DeleteRefund a df:DeleteAction ;
    df:writeOperation "delete" ;
    df:targetEntity ex:RefundRequest ;
    df:requiresHITL true .

ex:RefundRequest df:hasAction ex:CreateRefund , ex:DeleteRefund .
ex:Order df:hasAction ex:UpdateOrder .

# ---- Relationships ----
ex:RefundRequest df:relatedEntity ex:Order .
ex:RefundRequest df:relatedEntity ex:Customer .
ex:CustomerRisk df:relatedEntity ex:Customer .
"""


class TestCompileRefundOntology:
    """End-to-end extraction over the refund hero case."""

    @pytest.fixture
    def compiled(self) -> CompiledOntology:
        return compile_ontology(REFUND_OWL)

    def test_returns_compiled_ontology(self, compiled: CompiledOntology) -> None:
        assert isinstance(compiled, CompiledOntology)
        assert not compiled.is_empty()

    def test_writable_entities_extracted(self, compiled: CompiledOntology) -> None:
        assert set(compiled.entity_access.keys()) == {
            "RefundRequest",
            "Order",
            "CustomerRisk",
        }

    def test_read_only_entity_not_writable(self, compiled: CompiledOntology) -> None:
        # Customer is df:ReadableEntity only -> never in entity_access.
        assert "Customer" not in compiled.entity_access

    def test_allowed_operations_from_actions(self, compiled: CompiledOntology) -> None:
        # RefundRequest has an insert action and a delete action.
        assert compiled.entity_access["RefundRequest"] == {"insert", "delete"}
        assert compiled.entity_access["Order"] == {"update"}
        # CustomerRisk is writable but has no action -> empty op set.
        assert compiled.entity_access["CustomerRisk"] == set()

    def test_measure_field_additive(self, compiled: CompiledOntology) -> None:
        assert compiled.measure_fields["RefundRequest.ApprovedAmount"] == "additive"
        assert compiled.measure_fields["CustomerRisk.RiskScore"] == "additive"

    def test_state_field_choiceset(self, compiled: CompiledOntology) -> None:
        assert compiled.state_fields["Order.Status"] == "OrderStatusChoiceSet"

    def test_reference_field_target(self, compiled: CompiledOntology) -> None:
        assert compiled.reference_fields["RefundRequest.OrderId"] == "Order"

    def test_hitl_on_destructive_op(self, compiled: CompiledOntology) -> None:
        # DeleteRefund requires HITL -> delete op flagged on RefundRequest.
        assert compiled.hitl_operations.get("RefundRequest") == {"delete"}
        # Non-destructive ops are not flagged.
        assert "Order" not in compiled.hitl_operations

    def test_entity_relationships(self, compiled: CompiledOntology) -> None:
        rels = compiled.entity_relationships["RefundRequest"]
        assert set(rels) == {"Order", "Customer"}
        assert compiled.entity_relationships["CustomerRisk"] == ["Customer"]


class TestRfcDialect:
    """The RFC §4.1 dialect: rdf:type df:WritableEntity + df:allowsOperation."""

    def test_allows_operation_dialect(self) -> None:
        owl = """
        @prefix df: <https://ontology.uipath.com/datafabric#> .
        @prefix ex: <https://ontology.example.com/cc#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .

        ex:Contact a owl:Class, df:WritableEntity ;
            df:entityKey "Contact" ;
            df:allowsOperation "update" .

        ex:RefundRequest a owl:Class, df:WritableEntity ;
            df:entityKey "RefundRequest" ;
            df:allowsOperation "insert" .
        """
        compiled = compile_ontology(owl)
        assert compiled.entity_access["Contact"] == {"update"}
        assert compiled.entity_access["RefundRequest"] == {"insert"}

    def test_field_binding_via_local_name_fallback(self) -> None:
        # No df:hasField -> compiler infers owner from field_<Entity>_ name.
        owl = """
        @prefix df: <https://ontology.uipath.com/datafabric#> .
        @prefix ex: <https://ontology.example.com/cc#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .

        ex:CustomerRisk a owl:Class, df:WritableEntity ;
            df:entityKey "CustomerRisk" ;
            df:allowsOperation "update" .

        ex:field_CustomerRisk_Score a df:MeasureField ;
            df:fieldKey "RiskScore" ;
            df:measureSemantics "additive" .
        """
        compiled = compile_ontology(owl)
        assert compiled.measure_fields["CustomerRisk.RiskScore"] == "additive"


class TestGracefulPaths:
    """Empty / partial / malformed inputs."""

    def test_empty_string_returns_empty_ontology(self) -> None:
        compiled = compile_ontology("")
        assert isinstance(compiled, CompiledOntology)
        assert compiled.is_empty()

    def test_whitespace_returns_empty_ontology(self) -> None:
        assert compile_ontology("   \n  ").is_empty()

    def test_prefixes_only_returns_empty(self) -> None:
        owl = "@prefix df: <https://ontology.uipath.com/datafabric#> .\n"
        assert compile_ontology(owl).is_empty()

    def test_partial_ontology_extracts_what_is_present(self) -> None:
        # Only one writable entity, no fields/actions/relationships.
        owl = """
        @prefix df: <https://ontology.uipath.com/datafabric#> .
        @prefix ex: <https://ontology.example.com/cc#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .

        ex:Order a owl:Class, df:WritableEntity ;
            df:entityKey "Order" .
        """
        compiled = compile_ontology(owl)
        assert compiled.entity_access == {"Order": set()}
        assert compiled.measure_fields == {}
        assert compiled.entity_relationships == {}

    def test_malformed_turtle_raises_compile_error(self) -> None:
        owl = "this is not valid turtle <<< @@@ ;;;"
        with pytest.raises(OntologyCompileError):
            compile_ontology(owl)

    def test_measure_field_without_semantics_defaults_replacement(self) -> None:
        owl = """
        @prefix df: <https://ontology.uipath.com/datafabric#> .
        @prefix ex: <https://ontology.example.com/cc#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .

        ex:Acct a owl:Class, df:WritableEntity ; df:entityKey "Acct" .
        ex:field_Acct_Bal a df:MeasureField ; df:fieldKey "Balance" .
        ex:Acct df:hasField ex:field_Acct_Bal .
        """
        compiled = compile_ontology(owl)
        assert compiled.measure_fields["Acct.Balance"] == "replacement"


# The order-management domain, mirroring p1-owl-write-extension.ttl exactly
# (subClassOf dialect, action-derived ops, df:hasField, df:relatedEntity).
# Self-contained so the test does not depend on the sibling df-agent-os repo.
ORDER_OWL = """
@prefix df:   <https://ontology.uipath.com/datafabric#> .
@prefix ex:   <https://ontology.example.com/orders#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Customer a owl:Class ; rdfs:subClassOf df:ReadableEntity ; df:entityKey "Customer" .
ex:Product  a owl:Class ; rdfs:subClassOf df:ReadableEntity ; df:entityKey "Product" .
ex:Order    a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "Order" .
ex:OrderItem a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "OrderItem" .

ex:field_Order_Status a df:StateField ;
    df:fieldKey "Status" ; df:governedBy ex:OrderStatusMachine .
ex:field_Order_CustomerId a df:ReferenceField ;
    df:fieldKey "CustomerId" ; df:referencesEntity ex:Customer .
ex:field_OrderItem_OrderId a df:ReferenceField ;
    df:fieldKey "OrderId" ; df:referencesEntity ex:Order .
ex:field_OrderItem_ProductId a df:ReferenceField ;
    df:fieldKey "ProductId" ; df:referencesEntity ex:Product .

ex:Order df:hasField ex:field_Order_Status , ex:field_Order_CustomerId .
ex:OrderItem df:hasField ex:field_OrderItem_OrderId , ex:field_OrderItem_ProductId .

ex:CreateOrder a df:InsertAction ; df:writeOperation "insert" ; df:targetEntity ex:Order .
ex:UpdateOrderStatus a df:UpdateAction ; df:writeOperation "update" ;
    df:targetEntity ex:Order ; df:requiresHITL true .
ex:AddOrderItem a df:InsertAction ; df:writeOperation "insert" ; df:targetEntity ex:OrderItem .
ex:DeleteOrderItem a df:DeleteAction ; df:writeOperation "delete" ;
    df:targetEntity ex:OrderItem ; df:requiresHITL true .

ex:Order df:hasAction ex:CreateOrder , ex:UpdateOrderStatus .
ex:OrderItem df:hasAction ex:AddOrderItem , ex:DeleteOrderItem .

ex:Order df:relatedEntity ex:Customer , ex:OrderItem .
ex:OrderItem df:relatedEntity ex:Order , ex:Product .
"""


def test_compiles_order_management_dialect():
    """The order-management example (mirrors p1-owl-write-extension.ttl) compiles.

    Guards the .ttl dialect: rdfs:subClassOf df:WritableEntity, action-derived
    operations, df:hasField binding, df:governedBy state machines, and
    df:relatedEntity relationships. Read-only entities (Customer, Product)
    must be excluded from the writable set.
    """
    compiled = compile_ontology(ORDER_OWL)

    assert not compiled.is_empty()
    assert compiled.entity_access["Order"] == {"insert", "update"}
    assert compiled.entity_access["OrderItem"] == {"insert", "delete"}
    assert "Customer" not in compiled.entity_access
    assert "Product" not in compiled.entity_access
    assert compiled.hitl_operations["Order"] == {"update"}
    assert compiled.hitl_operations["OrderItem"] == {"delete"}
    assert compiled.state_fields["Order.Status"] == "OrderStatusMachine"
    assert compiled.reference_fields["Order.CustomerId"] == "Customer"
    assert compiled.reference_fields["OrderItem.OrderId"] == "Order"
    assert compiled.reference_fields["OrderItem.ProductId"] == "Product"
