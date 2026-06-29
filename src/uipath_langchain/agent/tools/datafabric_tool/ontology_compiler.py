"""Compile an OWL 2 QL Turtle ontology into a ``CompiledOntology``.

The OWL source is the *authoring and storage* format (see
``p1-owl-write-extension.ttl``).  It is NOT the prompt format — it is
compiled here into a structured ``CompiledOntology`` that the write
validator and tool-description builder consume.

Two ontology dialects are supported, because the RFC prose
(``p1-write-rfc-v2-ontology.md`` §4.1) and the shipped ``df:`` vocabulary
(``p1-owl-write-extension.ttl``) differ slightly:

  Entity access modes
    - .ttl dialect: ``ex:Order rdfs:subClassOf df:WritableEntity`` and the
      allowed operations come from the entity's actions
      (``df:hasAction`` -> action ``df:writeOperation``).
    - RFC dialect: ``ex:Order a df:WritableEntity ; df:allowsOperation "update"``.
    Both are extracted.

  Field semantics
    - Fields are individuals typed ``df:StateField`` / ``df:MeasureField`` /
      ``df:ReferenceField`` / ``df:ForeignKeyField`` with a ``df:fieldKey``.
    - Fields are bound to their owning entity via ``ex:Entity df:hasField
      ex:field_...``.  When a field is not bound via ``df:hasField`` (RFC
      dialect omits it) we fall back to matching the field individual's
      local name (``field_<Entity>_<...>``) against entity local names.

  HITL
    - ``action df:requiresHITL true`` with ``df:targetEntity`` +
      ``df:writeOperation`` -> ``hitl_operations[entity] = {op}``.

  Relationships
    - ``ex:A df:relatedEntity ex:B`` (.ttl) or ``ex:A df:referencesEntity
      ex:B`` between two entities (RFC) -> ``entity_relationships``.

The compiler is resilient to missing or partial annotations: it extracts
only what is present and never raises on a well-formed-but-sparse ontology.
A parse error on malformed Turtle is surfaced as ``OntologyCompileError``.
"""

from __future__ import annotations

import logging

from rdflib import RDF, RDFS, Graph, URIRef
from rdflib.term import Literal, Node

from .compiled_ontology import CompiledOntology

logger = logging.getLogger(__name__)

DF = "https://ontology.uipath.com/datafabric#"


class OntologyCompileError(ValueError):
    """Raised when the OWL Turtle source cannot be parsed."""


# df: vocabulary terms ------------------------------------------------------

_WRITABLE_ENTITY = URIRef(DF + "WritableEntity")
_STATE_FIELD = URIRef(DF + "StateField")
_MEASURE_FIELD = URIRef(DF + "MeasureField")
_REFERENCE_FIELD = URIRef(DF + "ReferenceField")
_FOREIGN_KEY_FIELD = URIRef(DF + "ForeignKeyField")

_ENTITY_KEY = URIRef(DF + "entityKey")
_FIELD_KEY = URIRef(DF + "fieldKey")
_HAS_FIELD = URIRef(DF + "hasField")
_HAS_ACTION = URIRef(DF + "hasAction")
_ALLOWS_OPERATION = URIRef(DF + "allowsOperation")
_WRITE_OPERATION = URIRef(DF + "writeOperation")
_TARGET_ENTITY = URIRef(DF + "targetEntity")
_REQUIRES_HITL = URIRef(DF + "requiresHITL")
_MEASURE_SEMANTICS = URIRef(DF + "measureSemantics")
_CHOICESET_KEY = URIRef(DF + "choiceSetKey")
_GOVERNED_BY = URIRef(DF + "governedBy")
_REFERENCES_ENTITY = URIRef(DF + "referencesEntity")
_RELATED_ENTITY = URIRef(DF + "relatedEntity")

_VALID_OPS = {"insert", "update", "delete"}


def _local_name(node: Node) -> str:
    """Return the fragment / last path segment of a URI (e.g. ``ex:Order`` -> ``Order``)."""
    text = str(node)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rsplit("/", 1)[-1]


def compile_ontology(owl_turtle: str) -> CompiledOntology:
    """Parse an OWL 2 QL Turtle ontology into a ``CompiledOntology``.

    Extracts entity access modes, field semantics (measure / state /
    reference), HITL markers, and entity relationships from the ``df:``
    vocabulary.  Resilient to missing or partial annotations — only
    extracts what is present.

    Args:
        owl_turtle: The OWL ontology serialised as Turtle.

    Returns:
        A ``CompiledOntology``.  Empty/whitespace input yields an empty
        ``CompiledOntology`` (``is_empty()`` is True).

    Raises:
        OntologyCompileError: if the Turtle source is malformed.
    """
    if not owl_turtle or not owl_turtle.strip():
        return CompiledOntology()

    graph = Graph()
    try:
        graph.parse(data=owl_turtle, format="turtle")
    except Exception as exc:  # rdflib raises a variety of parser exceptions
        raise OntologyCompileError(f"Failed to parse OWL Turtle: {exc}") from exc

    # 1. Map each entity *individual* (URIRef) -> its df:entityKey string.
    #    Entities are anything carrying df:entityKey (typed/subclassed as an
    #    entity by the vocabulary).  We key the compiled output by entityKey.
    entity_key_by_uri: dict[URIRef, str] = {}
    for subj, _pred, obj in graph.triples((None, _ENTITY_KEY, None)):
        if isinstance(subj, URIRef) and isinstance(obj, Literal):
            entity_key_by_uri[subj] = str(obj)

    # 2. Determine writable entities and their allowed operations.
    entity_access: dict[str, set[str]] = {}

    def _writable_uris() -> set[URIRef]:
        uris: set[URIRef] = set()
        # rdf:type df:WritableEntity (RFC dialect)
        for subj in graph.subjects(RDF.type, _WRITABLE_ENTITY):
            if isinstance(subj, URIRef):
                uris.add(subj)
        # rdfs:subClassOf df:WritableEntity (.ttl dialect)
        for subj in graph.subjects(RDFS.subClassOf, _WRITABLE_ENTITY):
            if isinstance(subj, URIRef):
                uris.add(subj)
        return uris

    for entity_uri in _writable_uris():
        key = entity_key_by_uri.get(entity_uri)
        if key is None:
            continue
        entity_access.setdefault(key, set())

    # 2a. Direct df:allowsOperation (RFC dialect).
    for subj, _pred, obj in graph.triples((None, _ALLOWS_OPERATION, None)):
        key = entity_key_by_uri.get(subj) if isinstance(subj, URIRef) else None
        if key is None:
            continue
        op = str(obj).strip().lower()
        if op in _VALID_OPS:
            entity_access.setdefault(key, set()).add(op)

    # 2b. Action-derived operations (.ttl dialect): an action declares a
    #     df:writeOperation and df:targetEntity; the entity may also bind the
    #     action via df:hasAction.  Build action -> (op, target entity key).
    action_op: dict[URIRef, str] = {}
    action_target_key: dict[URIRef, str] = {}
    for subj, _pred, obj in graph.triples((None, _WRITE_OPERATION, None)):
        if isinstance(subj, URIRef):
            op = str(obj).strip().lower()
            if op in _VALID_OPS:
                action_op[subj] = op
    for subj, _pred, obj in graph.triples((None, _TARGET_ENTITY, None)):
        if isinstance(subj, URIRef) and isinstance(obj, URIRef):
            target_key = entity_key_by_uri.get(obj)
            if target_key is not None:
                action_target_key[subj] = target_key
    # Also honour df:hasAction (entity -> action) as a target source.
    for entity_uri, _pred, action_uri in graph.triples((None, _HAS_ACTION, None)):
        if not (isinstance(entity_uri, URIRef) and isinstance(action_uri, URIRef)):
            continue
        key = entity_key_by_uri.get(entity_uri)
        if key is not None:
            action_target_key.setdefault(action_uri, key)

    for action_uri, op in action_op.items():
        target_key = action_target_key.get(action_uri)
        if target_key is not None:
            entity_access.setdefault(target_key, set()).add(op)

    # 3. HITL operations: action df:requiresHITL true -> hitl_operations.
    hitl_operations: dict[str, set[str]] = {}
    for subj, _pred, obj in graph.triples((None, _REQUIRES_HITL, None)):
        if not isinstance(subj, URIRef):
            continue
        if not (isinstance(obj, Literal) and bool(obj.toPython())):
            continue
        target_key = action_target_key.get(subj)
        op = action_op.get(subj)
        if target_key is not None and op is not None:
            hitl_operations.setdefault(target_key, set()).add(op)

    # 4. Field -> owning entity binding.
    #    Primary: ex:Entity df:hasField ex:field_...  (the .ttl dialect).
    field_entity_key: dict[URIRef, str] = {}
    for entity_uri, _pred, field_uri in graph.triples((None, _HAS_FIELD, None)):
        if not (isinstance(entity_uri, URIRef) and isinstance(field_uri, URIRef)):
            continue
        key = entity_key_by_uri.get(entity_uri)
        if key is not None:
            field_entity_key[field_uri] = key

    # Fallback: infer owning entity from the field's local name
    #   field_<EntityLocalName>_<...>  matched against entity local names.
    entity_localname_to_key = {
        _local_name(uri): key for uri, key in entity_key_by_uri.items()
    }

    def _owning_entity_key(field_uri: URIRef) -> str | None:
        if field_uri in field_entity_key:
            return field_entity_key[field_uri]
        local = _local_name(field_uri)
        if local.startswith("field_"):
            remainder = local[len("field_") :]
            # Greedily match the longest entity local-name prefix.
            best: str | None = None
            for ent_local in entity_localname_to_key:
                if remainder.startswith(ent_local + "_") or remainder == ent_local:
                    if best is None or len(ent_local) > len(best):
                        best = ent_local
            if best is not None:
                return entity_localname_to_key[best]
        return None

    def _field_key(field_uri: URIRef) -> str | None:
        val = graph.value(field_uri, _FIELD_KEY)
        return str(val) if val is not None else None

    def _compound_key(field_uri: URIRef) -> str | None:
        entity_key = _owning_entity_key(field_uri)
        field_key = _field_key(field_uri)
        if entity_key is None or field_key is None:
            return None
        return f"{entity_key}.{field_key}"

    # 5. Field semantics.
    measure_fields: dict[str, str] = {}
    state_fields: dict[str, str] = {}
    reference_fields: dict[str, str] = {}

    # Measure fields.
    for field_uri in graph.subjects(RDF.type, _MEASURE_FIELD):
        if not isinstance(field_uri, URIRef):
            continue
        compound = _compound_key(field_uri)
        if compound is None:
            continue
        semantics = graph.value(field_uri, _MEASURE_SEMANTICS)
        measure_fields[compound] = (
            str(semantics).strip().lower() if semantics is not None else "replacement"
        )

    # State fields -> choiceset / state-machine key.
    for field_uri in graph.subjects(RDF.type, _STATE_FIELD):
        if not isinstance(field_uri, URIRef):
            continue
        compound = _compound_key(field_uri)
        if compound is None:
            continue
        # Prefer df:choiceSetKey (RFC); fall back to df:governedBy (.ttl).
        cs = graph.value(field_uri, _CHOICESET_KEY)
        if cs is not None:
            state_fields[compound] = str(cs)
        else:
            gov = graph.value(field_uri, _GOVERNED_BY)
            state_fields[compound] = _local_name(gov) if isinstance(gov, URIRef) else ""

    # Reference / foreign-key fields -> referenced entity key.
    for ref_type in (_REFERENCE_FIELD, _FOREIGN_KEY_FIELD):
        for field_uri in graph.subjects(RDF.type, ref_type):
            if not isinstance(field_uri, URIRef):
                continue
            compound = _compound_key(field_uri)
            if compound is None:
                continue
            target = graph.value(field_uri, _REFERENCES_ENTITY)
            if isinstance(target, URIRef):
                target_key = entity_key_by_uri.get(target) or _local_name(target)
                reference_fields[compound] = target_key

    # 6. Entity-to-entity relationships (df:relatedEntity or df:referencesEntity
    #    where BOTH subject and object are entities).
    entity_relationships: dict[str, list[str]] = {}
    for pred in (_RELATED_ENTITY, _REFERENCES_ENTITY):
        for subj, _pred, obj in graph.triples((None, pred, None)):
            if not (isinstance(subj, URIRef) and isinstance(obj, URIRef)):
                continue
            subj_key = entity_key_by_uri.get(subj)
            obj_key = entity_key_by_uri.get(obj)
            # Only entity<->entity relationships (skip field->entity FK triples).
            if subj_key is None or obj_key is None:
                continue
            targets = entity_relationships.setdefault(subj_key, [])
            if obj_key not in targets:
                targets.append(obj_key)

    # Every entity carrying df:entityKey is "known" to the ontology — this
    # makes df:ReadableEntity first-class: a read-only entity is known but not
    # in entity_access (distinguishable from "unknown to the ontology").
    known_entities = set(entity_key_by_uri.values())

    compiled = CompiledOntology(
        known_entities=known_entities,
        entity_access=entity_access,
        measure_fields=measure_fields,
        state_fields=state_fields,
        reference_fields=reference_fields,
        hitl_operations=hitl_operations,
        entity_relationships=entity_relationships,
    )
    logger.debug(
        "Compiled ontology: %d known entities, %d writable, %d measure, "
        "%d state, %d reference fields, %d HITL entities",
        len(compiled.known_entities),
        len(compiled.entity_access),
        len(compiled.measure_fields),
        len(compiled.state_fields),
        len(compiled.reference_fields),
        len(compiled.hitl_operations),
    )
    return compiled


def format_ontology_debug(owl_turtle: str, compiled: CompiledOntology) -> str:
    """Render a debug block with both the raw OWL and the compiled IR.

    Returns a single string containing two clearly-headed sections: the raw
    OWL Turtle source and the human-readable compiled IR. Useful for debug
    logs and the ontology CLI scripts.

    Args:
        owl_turtle: The raw OWL Turtle source the ontology was compiled from.
        compiled: The compiled ontology IR.

    Returns:
        A multi-section debug string.
    """
    return (
        "=== RAW ONTOLOGY (OWL Turtle) ===\n"
        f"{owl_turtle.strip()}\n"
        "\n=== COMPILED ONTOLOGY (human-readable IR) ===\n"
        f"{compiled.to_human_readable()}"
    )


async def maybe_fetch_and_compile_ontology(
    entities_service: object,
) -> CompiledOntology | None:
    """Best-effort fetch + compile of the optional OWL ontology.

    Shared by both the Data Fabric read and write handlers. Looks for
    ``entities_service.get_ontology_file_async`` (which may only exist on a
    feature branch), fetches the OWL file, compiles it, and emits a debug log
    of the raw + compiled IR. Never raises: any absence/failure degrades to
    ``None`` (the metadata-only path).

    Args:
        entities_service: The resolved EntitiesService instance.

    Returns:
        The compiled ontology, or ``None`` when no ontology is available or
        the platform package does not expose the fetch method.
    """
    get_ontology = getattr(entities_service, "get_ontology_file_async", None)
    if not callable(get_ontology):
        logger.debug(
            "EntitiesService has no get_ontology_file_async; "
            "skipping ontology compilation (metadata-only)."
        )
        return None

    try:
        owl_turtle = await get_ontology("owl")
        if not owl_turtle:
            logger.debug("No OWL ontology returned; metadata-only.")
            return None
        compiled = compile_ontology(owl_turtle)
        logger.debug(format_ontology_debug(owl_turtle, compiled))
        return compiled
    except Exception as exc:  # graceful no-op on any fetch/parse failure
        logger.debug("Ontology fetch/compile skipped: %s", exc)
        return None
