"""Natural-language write schema builder for Data Fabric tool descriptions.

Converts resolved ``EntityWriteSchema`` objects into a token-efficient,
LLM-native tool description.  This replaces raw OWL injection — the
intermediate representation pattern from the RFC §5.6.

The generated description is consumed by the outer agent as the write
tool's ``description`` field, giving the LLM structured knowledge of
which entities are writable, their fields, types, constraints, and
allowed operations.
"""

from __future__ import annotations

from .models import EntityWriteSchema


def build_write_tool_description(
    write_schemas: dict[str, EntityWriteSchema],
    entity_access: dict[str, set[str]] | None = None,
) -> str:
    """Build a natural-language write tool description from resolved entity schemas.

    This replaces raw OWL injection.  The description is token-efficient and
    LLM-native, following the intermediate representation pattern from the RFC.

    Args:
        write_schemas: Mapping of entity_key -> EntityWriteSchema for writable
            entities.  These carry the field-level detail (name, type,
            required flag, choiceset indicator).
        entity_access: Optional mapping of entity_key -> set of allowed
            operations (e.g. ``{"insert", "update"}``).  When provided the
            description lists allowed ops per entity; otherwise all three
            operations are assumed available.

    Returns:
        A multi-line markdown-ish description string suitable for use as
        a LangChain tool ``description``.
    """
    if not write_schemas:
        return (
            "Modify Data Fabric entities using structured operations "
            "(insert, update, delete).\n\n"
            "No writable entities are currently configured."
        )

    lines: list[str] = [
        "Modify Data Fabric entities using structured operations "
        "(insert, update, delete).",
    ]

    all_ops = {"insert", "update", "delete"}

    for entity_key, schema in sorted(write_schemas.items()):
        ops = entity_access.get(entity_key, all_ops) if entity_access else all_ops
        ops_str = ", ".join(sorted(ops))
        lines.append(f"\n### {schema.display_name} ({ops_str})")

        for field in schema.writable_fields:
            parts: list[str] = [field.type_name.upper()]
            if field.is_required:
                parts.append("required")
            if field.is_choiceset:
                parts.append("CHOICE_SET")
            field_desc = f"  - {field.name} ({', '.join(parts)})"
            lines.append(field_desc)

    lines.append("")
    lines.append("Operations:")
    lines.append(
        "- insert: provide entity_key and fields. All required fields must be included."
    )
    lines.append(
        "- update: provide entity_key, record_id (from a prior read), "
        "and fields to change."
    )
    lines.append("- delete: provide entity_key and record_id. Requires confirmation.")
    lines.append("")
    lines.append(
        "Query the entity first (using the read tool) to discover record IDs "
        "and current field values before updating or deleting."
    )

    return "\n".join(lines)
