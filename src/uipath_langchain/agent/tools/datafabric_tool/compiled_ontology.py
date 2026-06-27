"""Compiled OWL ontology model for Data Fabric writes.

A ``CompiledOntology`` is the intermediate representation produced by
``ontology_compiler.compile_ontology`` from an OWL 2 QL Turtle source
(the ``df:`` write-extension vocabulary, see ``p1-owl-write-extension.ttl``).

The ontology is OPTIONAL.  When present it enriches and constrains writes
with semantics that entity metadata alone cannot express:

  - which entities are writable and which operations they allow
  - field semantics (state / measure / reference)
  - HITL markers on destructive operations
  - entity-to-entity relationships

When the ontology is absent the metadata-only write path still works
(graceful fallback).  See RFC ``p1-write-rfc-v2-ontology.md`` §5.2.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompiledOntology(BaseModel):
    """Result of compiling an OWL write-extension ontology.

    All members are keyed by the ``df:entityKey`` / ``df:fieldKey`` strings,
    which are the exact values the LLM uses in ``DataFabricWriteInput``.
    Field-level members are keyed as ``"<entity_key>.<field_key>"``.

    Every member defaults to empty, so a partial or empty ontology yields a
    valid (if sparse) ``CompiledOntology`` rather than raising.
    """

    entity_access: dict[str, set[str]] = Field(default_factory=dict)
    """entity_key -> set of allowed operations, e.g. ``{"insert", "update"}``."""

    measure_fields: dict[str, str] = Field(default_factory=dict)
    """``"entity_key.field_key"`` -> ``"additive"`` | ``"replacement"``."""

    state_fields: dict[str, str] = Field(default_factory=dict)
    """``"entity_key.field_key"`` -> choiceset / state-machine key."""

    reference_fields: dict[str, str] = Field(default_factory=dict)
    """``"entity_key.field_key"`` -> referenced entity_key (FK target)."""

    hitl_operations: dict[str, set[str]] = Field(default_factory=dict)
    """entity_key -> set of operations that require human-in-the-loop."""

    entity_relationships: dict[str, list[str]] = Field(default_factory=dict)
    """entity_key -> list of referenced entity_keys (semantic relationships)."""

    def is_empty(self) -> bool:
        """True if no ontology facts were extracted (graceful-fallback signal)."""
        return not (
            self.entity_access
            or self.measure_fields
            or self.state_fields
            or self.reference_fields
            or self.hitl_operations
            or self.entity_relationships
        )
