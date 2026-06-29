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

    known_entities: set[str] = Field(default_factory=set)
    """Every entity_key the ontology declares (has ``df:entityKey``), readable
    or writable. A read-only ``df:ReadableEntity`` is "known but not in
    ``entity_access``"; an entity absent here is unknown to the ontology."""

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
            self.known_entities
            or self.entity_access
            or self.measure_fields
            or self.state_fields
            or self.reference_fields
            or self.hitl_operations
            or self.entity_relationships
        )

    def is_known(self, entity_key: str) -> bool:
        """True if the ontology declares this entity (readable or writable)."""
        return entity_key in self.known_entities

    def is_writable(self, entity_key: str) -> bool:
        """True if the ontology grants any write access to this entity."""
        return entity_key in self.entity_access

    def is_read_only(self, entity_key: str) -> bool:
        """True if the entity is declared by the ontology but grants no writes.

        A ``df:ReadableEntity`` is "known but not in ``entity_access``".
        """
        return (
            entity_key in self.known_entities and entity_key not in self.entity_access
        )

    def to_human_readable(self) -> str:
        """Render a compact, grouped, human-readable summary of the IR.

        Sections: entity access modes (with HITL ops), field semantics
        (measure / state / reference), and entity relationships. Intended
        for debug logs and the ontology CLI scripts.
        """
        lines: list[str] = []

        # -- Entities (access mode + HITL) --
        lines.append("Entities:")
        if self.known_entities:
            for ek in sorted(self.known_entities):
                ops = self.entity_access.get(ek)
                if ops is not None:
                    mode = (
                        f"WRITABLE [{','.join(sorted(ops))}]"
                        if ops
                        else "WRITABLE [no ops declared]"
                    )
                else:
                    mode = "READ-ONLY"
                line = f"  - {ek}: {mode}"
                hitl = self.hitl_operations.get(ek)
                if hitl:
                    line += f" (HITL: {','.join(sorted(hitl))})"
                lines.append(line)
        else:
            lines.append("  (none)")

        # -- Field semantics --
        lines.append("Field semantics:")
        if self.measure_fields:
            lines.append("  Measure fields (additive / replacement):")
            for k in sorted(self.measure_fields):
                lines.append(f"    - {k}: {self.measure_fields[k]}")
        if self.state_fields:
            lines.append("  State fields (choiceset / state-machine):")
            for k in sorted(self.state_fields):
                src = self.state_fields[k] or "(unspecified)"
                lines.append(f"    - {k}: {src}")
        if self.reference_fields:
            lines.append("  Reference fields (-> target entity):")
            for k in sorted(self.reference_fields):
                lines.append(f"    - {k} -> {self.reference_fields[k]}")
        if not (self.measure_fields or self.state_fields or self.reference_fields):
            lines.append("  (none)")

        # -- Relationships --
        lines.append("Relationships:")
        if self.entity_relationships:
            for ek in sorted(self.entity_relationships):
                targets = ", ".join(self.entity_relationships[ek])
                lines.append(f"  - {ek} -> {targets}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)
