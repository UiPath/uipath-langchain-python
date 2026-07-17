"""Parse OWL Functional Notation into an in-memory graph topology.

The ontology-runtime stores schemas as OWL 2 Functional Notation (``.ofn``).
This module extracts the entity-relationship graph from the OWL text so the
EoG controller can traverse it deterministically — without relying on the LLM
to decide what to explore next.

The parsed graph is a lightweight adjacency structure:
- Nodes = OWL classes (entity types)
- Edges = OWL object properties with domain/range (typed, directed relationships)
- Data properties per entity (for context contract: what fields exist)
- Function-to-entity mapping (which functions touch which entities)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OntologyNode:
    """An entity type (OWL class) in the ontology graph."""

    name: str
    label: str = ""
    data_properties: list[DataProperty] = field(default_factory=list)


@dataclass(frozen=True)
class DataProperty:
    """A data property (column) on an entity."""

    name: str
    range: str = "xsd:string"


@dataclass(frozen=True)
class OntologyEdge:
    """A typed, directed relationship (OWL object property) between entities."""

    name: str
    label: str
    source: str  # domain entity
    target: str  # range entity


@dataclass
class OntologyGraph:
    """In-memory entity-relationship graph parsed from OWL.

    Provides adjacency lookups for the EoG controller:
    - ``neighbors(entity)`` — all entities connected by a relationship
    - ``outgoing(entity)`` / ``incoming(entity)`` — directed neighbors
    - ``edges_of(entity)`` — all edges involving an entity
    - ``functions_for(entity)`` — functions whose params or description
      indicate they operate on this entity type
    """

    nodes: dict[str, OntologyNode] = field(default_factory=dict)
    edges: list[OntologyEdge] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)

    # Pre-built adjacency (call _build_adjacency after construction)
    _outgoing: dict[str, list[OntologyEdge]] = field(default_factory=dict)
    _incoming: dict[str, list[OntologyEdge]] = field(default_factory=dict)
    _fn_by_entity: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )

    def _build_adjacency(self) -> None:
        """Build adjacency indexes from edges and function list."""
        self._outgoing = {n: [] for n in self.nodes}
        self._incoming = {n: [] for n in self.nodes}
        for edge in self.edges:
            if edge.source in self._outgoing:
                self._outgoing[edge.source].append(edge)
            if edge.target in self._incoming:
                self._incoming[edge.target].append(edge)

        # Map functions to entities by scanning params and statement text
        self._fn_by_entity = {n: [] for n in self.nodes}
        for fn in self.functions:
            touched = _infer_touched_entities(fn, set(self.nodes.keys()))
            for entity_name in touched:
                self._fn_by_entity[entity_name].append(fn)

    def neighbors(self, entity: str) -> list[str]:
        """All entities connected to ``entity`` by any edge (either direction)."""
        seen: set[str] = set()
        for edge in self._outgoing.get(entity, []):
            if edge.target != entity:
                seen.add(edge.target)
        for edge in self._incoming.get(entity, []):
            if edge.source != entity:
                seen.add(edge.source)
        return sorted(seen)

    def outgoing(self, entity: str) -> list[OntologyEdge]:
        """Edges where ``entity`` is the source (domain)."""
        return self._outgoing.get(entity, [])

    def incoming(self, entity: str) -> list[OntologyEdge]:
        """Edges where ``entity`` is the target (range)."""
        return self._incoming.get(entity, [])

    def edges_of(self, entity: str) -> list[OntologyEdge]:
        """All edges involving ``entity`` in either direction."""
        return self.outgoing(entity) + self.incoming(entity)

    def functions_for(self, entity: str) -> list[dict[str, Any]]:
        """Functions that touch this entity type."""
        return self._fn_by_entity.get(entity, [])

    def entity_for_id(self, entity_id: str) -> str | None:
        """Infer entity type from an instance ID using prefix conventions.

        E.g., ``"INV-2004"`` → ``"Invoice"`` if the entity's key property
        pattern matches. Falls back to checking if the ID starts with the
        entity name.
        """
        # Check common S2P prefixes
        for prefix, entity_name in _ID_PREFIXES.items():
            if entity_id.startswith(prefix) and entity_name in self.nodes:
                return entity_name
        # Fallback: check if entity_id starts with an entity name
        for name in self.nodes:
            if entity_id.startswith(name):
                return name
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage in LangGraph state."""
        return {
            "nodes": {
                name: {
                    "name": node.name,
                    "label": node.label,
                    "data_properties": [
                        {"name": p.name, "range": p.range}
                        for p in node.data_properties
                    ],
                }
                for name, node in self.nodes.items()
            },
            "edges": [
                {
                    "name": e.name,
                    "label": e.label,
                    "source": e.source,
                    "target": e.target,
                }
                for e in self.edges
            ],
            "functions": self.functions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OntologyGraph:
        """Reconstruct from serialized dict (e.g., from LangGraph state)."""
        nodes: dict[str, OntologyNode] = {}
        for name, ndata in data.get("nodes", {}).items():
            props = [
                DataProperty(name=p["name"], range=p.get("range", "xsd:string"))
                for p in ndata.get("data_properties", [])
            ]
            nodes[name] = OntologyNode(
                name=ndata["name"],
                label=ndata.get("label", ""),
                data_properties=props,
            )

        edges = [
            OntologyEdge(
                name=e["name"],
                label=e.get("label", e["name"]),
                source=e["source"],
                target=e["target"],
            )
            for e in data.get("edges", [])
        ]

        graph = cls(
            nodes=nodes,
            edges=edges,
            functions=data.get("functions", []),
        )
        graph._build_adjacency()
        return graph


# ── OWL Functional Notation parser ──────────────────────────────────

_CLASS_RE = re.compile(r"Declaration\(Class\(:(\w+)\)\)")
_OBJ_PROP_RE = re.compile(r"Declaration\(ObjectProperty\(:(\w+)\)\)")
_DATA_PROP_RE = re.compile(r"Declaration\(DataProperty\(:(\w+)\.(\w+)\)\)")
_OBJ_DOMAIN_RE = re.compile(r"ObjectPropertyDomain\(:(\w+)\s+:(\w+)\)")
_OBJ_RANGE_RE = re.compile(r"ObjectPropertyRange\(:(\w+)\s+:(\w+)\)")
_DATA_DOMAIN_RE = re.compile(r"DataPropertyDomain\(:(\w+)\.(\w+)\s+:(\w+)\)")
_DATA_RANGE_RE = re.compile(r"DataPropertyRange\(:(\w+)\.(\w+)\s+(xsd:\w+)\)")
_LABEL_RE = re.compile(
    r'AnnotationAssertion\(rdfs:label :(\w+(?:\.\w+)?)\s+"([^"]+)"\)'
)


def parse_ofn(ofn_text: str) -> OntologyGraph:
    """Parse OWL Functional Notation into an OntologyGraph.

    Extracts classes, object properties (with domain/range), data properties,
    and annotation labels. Does not require rdflib or OWLAPI — pure regex
    parsing of the OFN text format.

    Args:
        ofn_text: The ``.ofn`` file content.

    Returns:
        An ``OntologyGraph`` with nodes, edges, and adjacency built.
    """
    # 1. Extract classes
    class_names = _CLASS_RE.findall(ofn_text)

    # 2. Extract labels
    labels: dict[str, str] = {}
    for name, label in _LABEL_RE.findall(ofn_text):
        labels[name] = label

    # 3. Extract data properties with domain + range
    dp_domains: dict[str, list[tuple[str, str]]] = {}  # entity → [(prop, range)]
    data_ranges: dict[str, str] = {}  # "Entity.prop" → xsd:type
    for entity, prop, xsd_type in _DATA_RANGE_RE.findall(ofn_text):
        data_ranges[f"{entity}.{prop}"] = xsd_type
    for entity, prop, domain_entity in _DATA_DOMAIN_RE.findall(ofn_text):
        range_type = data_ranges.get(f"{entity}.{prop}", "xsd:string")
        if domain_entity not in dp_domains:
            dp_domains[domain_entity] = []
        dp_domains[domain_entity].append((prop, range_type))

    # 4. Build nodes
    nodes: dict[str, OntologyNode] = {}
    for class_name in class_names:
        props = [
            DataProperty(name=p, range=r)
            for p, r in dp_domains.get(class_name, [])
        ]
        nodes[class_name] = OntologyNode(
            name=class_name,
            label=labels.get(class_name, class_name),
            data_properties=props,
        )

    # 5. Extract object properties with domain/range → edges
    obj_domains: dict[str, str] = {}
    obj_ranges: dict[str, str] = {}
    for prop_name, domain_entity in _OBJ_DOMAIN_RE.findall(ofn_text):
        obj_domains[prop_name] = domain_entity
    for prop_name, range_entity in _OBJ_RANGE_RE.findall(ofn_text):
        obj_ranges[prop_name] = range_entity

    edges: list[OntologyEdge] = []
    for prop_name in _OBJ_PROP_RE.findall(ofn_text):
        source = obj_domains.get(prop_name)
        target = obj_ranges.get(prop_name)
        if source and target:
            edges.append(OntologyEdge(
                name=prop_name,
                label=labels.get(prop_name, prop_name),
                source=source,
                target=target,
            ))

    graph = OntologyGraph(nodes=nodes, edges=edges)
    graph._build_adjacency()
    return graph


# ── Function-to-entity inference ────────────────────────────────────

def _infer_touched_entities(
    fn: dict[str, Any], entity_names: set[str]
) -> set[str]:
    """Infer which entities a function touches.

    Checks:
    1. Function params whose names match an entity's key property pattern
       (e.g., ``invoiceId`` → ``Invoice``).
    2. SPARQL statement referencing ``ont:EntityName`` or ``ont:EntityName.prop``.
    3. Function label/description mentioning entity names.
    """
    touched: set[str] = set()
    params = fn.get("params", [])
    statement = fn.get("statement", "")
    label = fn.get("label", "")

    # From params: "invoiceId" → "Invoice", "supplierId" → "Supplier"
    for param in params:
        pname = param.get("name", "")
        for entity_name in entity_names:
            # Match: param ends with entity's conventional key suffix
            # e.g., invoiceId → Invoice, poId → PurchaseOrder
            if pname.lower() == (entity_name.lower() + "id") or \
               pname.lower() == (entity_name.lower() + "_id"):
                touched.add(entity_name)

    # From SPARQL statement: look for "ont:EntityName" or ":EntityName"
    for entity_name in entity_names:
        if f":{entity_name}" in statement or \
           f"ont:{entity_name}" in statement:
            touched.add(entity_name)

    # From label: case-insensitive entity name match
    label_lower = label.lower()
    for entity_name in entity_names:
        if entity_name.lower() in label_lower:
            touched.add(entity_name)

    return touched


# ── ID prefix convention ────────────────────────────────────────────

_ID_PREFIXES: dict[str, str] = {
    "INV-": "Invoice",
    "SUP-": "Supplier",
    "PO-": "PurchaseOrder",
    "EXC-": "ToleranceException",
    "COM-": "Commodity",
    "RULE-": "ExceptionRule",
    "CTR-": "Contract",
    "PR-": "Requisition",
    "SPD-": "SpendRecord",
    "FX-": "FXRate",
    "SCP-": "ContractScope",
    "CC-": "CostCenter",
    "BUD-": "Budget",
    "ITM-": "Item",
}
