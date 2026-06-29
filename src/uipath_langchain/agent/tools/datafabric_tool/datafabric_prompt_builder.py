"""Schema context building and formatting for Data Fabric entities.

Converts raw Entity SDK objects into structured Pydantic models (SQLContext),
then formats them as text for system prompt injection.

The SQL strategy section (``sql_expert_system_prompt``) is rendered from a
versioned prompt template via the ``prompts`` package. ``SQL_CONSTRAINTS`` is
appended verbatim — the system prompt should describe strategy only, not
backend deny-lists.
"""

import logging

from uipath.platform.entities import Entity

from .compiled_ontology import CompiledOntology
from .datafabric_prompts import SQL_CONSTRAINTS
from .models import (
    EntitySchema,
    EntitySQLContext,
    EntityWriteSchema,
    FieldSchema,
    QueryPattern,
    SQLContext,
)
from .prompts import build_prompt_context, get_prompt_version
from .write_schema_builder import build_write_tool_description
from .write_validation import derive_writable_fields, is_entity_writable

logger = logging.getLogger(__name__)


def build_entity_context(entity: Entity) -> EntitySQLContext:
    """Convert an Entity SDK object to schema + derived query patterns."""
    field_schemas: list[FieldSchema] = []
    numeric_field: str | None = None
    text_field: str | None = None
    pk_field: str | None = None

    # System fields (CreateTime, UpdatedBy, ...) are analytical noise and stay
    # hidden. The one exception is the primary key on a *writable* entity: the
    # agent needs it to retrieve record ids (the record_id for update/delete)
    # and as a stable column to ORDER BY. Read-only entities keep all system
    # fields hidden.
    writable = is_entity_writable(entity)
    seen_names: set[str] = set()

    for field in entity.fields or []:
        is_pk = bool(getattr(field, "is_primary_key", False))
        if field.is_hidden_field:
            continue
        if field.is_system_field and not (writable and is_pk):
            continue
        # P3 collision guard: when a user/CSV field shares a system field's
        # name (e.g. an imported "Id"), keep the first occurrence rather than
        # emitting a duplicate column row. Full disambiguation is P3 work.
        if field.name in seen_names:
            continue
        seen_names.add(field.name)

        type_name = field.sql_type.name if field.sql_type else "unknown"
        fs = FieldSchema(
            name=field.name,
            display_name=field.display_name,
            type=type_name,
            description=field.description,
            is_foreign_key=field.is_foreign_key,
            is_required=field.is_required,
            is_unique=field.is_unique,
            nullable=not field.is_required,
        )
        field_schemas.append(fs)

        if is_pk and writable and pk_field is None:
            pk_field = fs.name
        if not numeric_field and fs.is_numeric:
            numeric_field = fs.name
        if not text_field and fs.is_text:
            text_field = fs.name

    field_names = [f.name for f in field_schemas]
    table = entity.name

    group_field = text_field or (field_names[0] if field_names else "Category")
    agg_field = numeric_field or (field_names[1] if len(field_names) > 1 else "Amount")
    filter_field = text_field or (field_names[0] if field_names else "Name")
    count_col = pk_field or (field_names[0] if field_names else "id")

    # Put the primary key first in the projection so record-level reads return
    # the id the agent reuses as record_id when writing.
    if pk_field:
        ordered = [pk_field] + [n for n in field_names if n != pk_field]
    else:
        ordered = field_names
    fields_sample = ", ".join(ordered[:5]) if ordered else "*"
    # A stable sort column for paginated reads. Prefer the primary key; this
    # prevents the SQL-gen model from falling back to a non-existent pseudo
    # column (e.g. "rowid") when a query needs ORDER BY + LIMIT.
    stable_sort = pk_field or (field_names[0] if field_names else None)

    query_patterns = [
        QueryPattern(
            intent="Show all",
            sql=(
                f"SELECT {fields_sample} FROM {table} ORDER BY {stable_sort} LIMIT 100"
                if stable_sort
                else f"SELECT {fields_sample} FROM {table} LIMIT 100"
            ),
        ),
        QueryPattern(
            intent="Find by X",
            sql=f"SELECT {fields_sample} FROM {table} WHERE {filter_field} = 'value' LIMIT 100",
        ),
        QueryPattern(
            intent="Top N by Y",
            sql=f"SELECT {fields_sample} FROM {table} ORDER BY {agg_field} DESC LIMIT N",
        ),
        QueryPattern(
            intent="Count by X",
            sql=f"SELECT {group_field}, COUNT({count_col}) as count FROM {table} GROUP BY {group_field}",
        ),
        QueryPattern(
            intent="Top N segments",
            sql=f"SELECT {group_field}, COUNT({count_col}) as count FROM {table} GROUP BY {group_field} ORDER BY count DESC LIMIT N",
        ),
        QueryPattern(
            intent="Sum/Avg of Y",
            sql=f"SELECT SUM({agg_field}) as total FROM {table}",
        ),
    ]
    # For writable entities, give the model an explicit record-lookup pattern so
    # it knows how to fetch a single row's id before an update/delete.
    if pk_field:
        query_patterns.insert(
            1,
            QueryPattern(
                intent="Get a record's id to update/delete it",
                sql=f"SELECT {fields_sample} FROM {table} WHERE {filter_field} = 'value' LIMIT 1",
            ),
        )

    schema = EntitySchema(
        id=entity.id,
        entity_name=entity.name,
        display_name=entity.display_name or entity.name,
        description=entity.description,
        record_count=entity.record_count,
        fields=field_schemas,
    )
    return EntitySQLContext(entity_schema=schema, query_patterns=query_patterns)


def format_ontology_context(compiled_ontology: CompiledOntology) -> str:
    """Render read-side schema-linking enrichment from a compiled ontology.

    This is informational context for the NL-to-SQL model (the P5 goal). It
    surfaces per-entity access modes, entity relationships, reference/FK fields
    (to guide join-path selection), and state fields (with their valid-value
    source). It does NOT restrict reads.

    Args:
        compiled_ontology: The compiled OWL ontology IR.

    Returns:
        A markdown ``## Ontology Context`` section, or an empty string when the
        ontology carries no facts.
    """
    if compiled_ontology is None or compiled_ontology.is_empty():
        return ""

    lines: list[str] = ["## Ontology Context", ""]
    lines.append(
        "Ontology-derived schema-linking hints (informational; does not "
        "restrict what you may read):"
    )
    lines.append("")

    # Per-entity access mode (purely informational for the read model).
    if compiled_ontology.known_entities:
        lines.append("**Entity access modes:**")
        for ek in sorted(compiled_ontology.known_entities):
            mode = "WRITABLE" if compiled_ontology.is_writable(ek) else "READ-ONLY"
            lines.append(f"- {ek}: {mode}")
        lines.append("")

    # Entity relationships (entity -> related entities).
    if compiled_ontology.entity_relationships:
        lines.append("**Entity relationships (entity -> related entities):**")
        for ek in sorted(compiled_ontology.entity_relationships):
            targets = ", ".join(compiled_ontology.entity_relationships[ek])
            lines.append(f"- {ek} -> {targets}")
        lines.append("")

    # Reference / FK fields (guide join-path selection).
    if compiled_ontology.reference_fields:
        lines.append("**Reference / foreign-key fields (field -> target entity):**")
        for k in sorted(compiled_ontology.reference_fields):
            lines.append(f"- {k} -> {compiled_ontology.reference_fields[k]}")
        lines.append("")

    # State fields and their valid-value source.
    if compiled_ontology.state_fields:
        lines.append("**State fields (field -> valid-value source):**")
        for k in sorted(compiled_ontology.state_fields):
            src = compiled_ontology.state_fields[k] or "(unspecified)"
            lines.append(f"- {k} -> {src}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_sql_context(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
    prompt_version: str | None = None,
    compiled_ontology: CompiledOntology | None = None,
) -> SQLContext:
    """Build the full SQL context from entities, prompts, and constraints.

    Args:
        entities: Resolved Data Fabric entities.
        resource_description: Optional free-text description folded into the
            rendered prompt as ``## Domain Guidance``.
        base_system_prompt: Optional outer-agent system prompt prepended as
            ``## Agent Instructions``.
        prompt_version: Optional version key (e.g. ``"v0"``, ``"v1"``).
            Defaults to the registry's default.
    """
    version = get_prompt_version(prompt_version)
    ctx = build_prompt_context(
        entities=entities,
        resource_description=resource_description,
    )
    rendered_prompt = version.render(ctx)

    ontology_context = (
        format_ontology_context(compiled_ontology)
        if compiled_ontology is not None
        else ""
    )

    return SQLContext(
        base_system_prompt=base_system_prompt or None,
        resource_description=None,
        sql_expert_system_prompt=rendered_prompt,
        constraints=SQL_CONSTRAINTS,
        ontology_context=ontology_context or None,
        entity_contexts=[build_entity_context(e) for e in entities],
    )


def format_sql_context(ctx: SQLContext) -> str:
    """Format a SQLContext as text for system prompt injection."""
    lines: list[str] = []

    if ctx.base_system_prompt:
        lines.append("## Agent Instructions")
        lines.append("")
        lines.append(ctx.base_system_prompt)
        lines.append("")

    if ctx.sql_expert_system_prompt:
        lines.append("## SQL Query Generation Guidelines")
        lines.append("")
        lines.append(ctx.sql_expert_system_prompt)
        lines.append("")

    if ctx.constraints:
        lines.append("## SQL Constraints")
        lines.append("")
        lines.append(ctx.constraints)
        lines.append("")

    if ctx.resource_description:
        lines.append("## Entity set description")
        lines.append("")
        lines.append(ctx.resource_description)
        lines.append("")

    if ctx.ontology_context:
        lines.append(ctx.ontology_context.rstrip())
        lines.append("")

    lines.append("## All available Data Fabric Entities")
    lines.append("")

    for entity_ctx in ctx.entity_contexts:
        entity = entity_ctx.entity_schema
        lines.append(
            f"### Entity: {entity.display_name} (SQL table: `{entity.entity_name}`)"
        )
        if entity.description:
            lines.append(f"_{entity.description}_")
        lines.append("")
        lines.append("| Field | Type |")
        lines.append("|-------|------|")

        for field in entity.fields:
            lines.append(f"| {field.name} | {field.display_type} |")

        lines.append("")

        lines.append(f"**Query Patterns for {entity.entity_name}:**")
        lines.append("")
        lines.append("| User Intent | SQL Pattern |")
        lines.append("|-------------|-------------|")
        for p in entity_ctx.query_patterns:
            lines.append(f"| '{p.intent}' | `{p.sql}` |")
        lines.append("")

    return "\n".join(lines)


def build(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
    prompt_version: str | None = None,
    compiled_ontology: CompiledOntology | None = None,
) -> str:
    """Build the full SQL prompt text for the inner sub-graph LLM.

    Combines agent system prompt, the rendered SQL strategy prompt, the
    Calcite constraint deny-list, and entity schemas + query patterns.

    Args:
        entities: List of Entity objects with schema information.
        resource_description: Optional description of the resource/entity set;
            folded into the rendered prompt as domain guidance.
        base_system_prompt: Optional system prompt from the outer agent.
        prompt_version: Optional version key (e.g. ``"v0"``, ``"v1"``).
            Defaults to the registry's default.
        compiled_ontology: Optional compiled OWL ontology. When present an
            ``## Ontology Context`` section is appended with read-side
            schema-linking enrichment (relationships, FK targets, state-value
            sources, access modes). Purely informational — never restricts reads.

    Returns:
        Formatted prompt string for the inner LLM system message.
    """
    if not entities:
        return ""

    ctx = build_sql_context(
        entities,
        resource_description,
        base_system_prompt,
        prompt_version=prompt_version,
        compiled_ontology=compiled_ontology,
    )
    return format_sql_context(ctx)


def build_write_context(entities: list[Entity]) -> str:
    """Build write-relevant schema context for the system prompt.

    Generates a natural-language description of writable entities, their
    fields with types and constraints, ChoiceSet indicators, and allowed
    operations.  This is appended to the outer agent's system prompt so
    the LLM knows which entities can be written to and how.

    Args:
        entities: Resolved Entity objects from the platform.

    Returns:
        Formatted markdown string describing writable entities and
        their schemas, or an empty string if no entities are writable.
    """
    write_schemas: dict[str, EntityWriteSchema] = {}
    for entity in entities:
        if not is_entity_writable(entity):
            continue
        writable_fields = derive_writable_fields(entity)
        if writable_fields:
            write_schemas[entity.name] = EntityWriteSchema(
                entity_key=entity.name,
                display_name=entity.display_name or entity.name,
                writable_fields=writable_fields,
            )

    if not write_schemas:
        return ""

    lines: list[str] = [
        "## Writable Data Fabric Entities",
        "",
        build_write_tool_description(write_schemas),
    ]
    return "\n".join(lines)
