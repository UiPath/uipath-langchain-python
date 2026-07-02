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

from .datafabric_prompts import SQL_CONSTRAINTS
from .models import (
    EntitySchema,
    EntitySQLContext,
    FieldSchema,
    QueryPattern,
    SQLContext,
)
from .prompts import build_prompt_context, get_prompt_version

logger = logging.getLogger(__name__)


def build_entity_context(entity: Entity) -> EntitySQLContext:
    """Convert an Entity SDK object to schema + derived query patterns."""
    field_schemas: list[FieldSchema] = []
    numeric_field: str | None = None
    text_field: str | None = None

    for field in entity.fields or []:
        if field.is_hidden_field or field.is_system_field:
            continue
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

        if not numeric_field and fs.is_numeric:
            numeric_field = fs.name
        if not text_field and fs.is_text:
            text_field = fs.name

    field_names = [f.name for f in field_schemas]
    table = entity.name

    group_field = text_field or (field_names[0] if field_names else "Category")
    agg_field = numeric_field or (field_names[1] if len(field_names) > 1 else "Amount")
    filter_field = text_field or (field_names[0] if field_names else "Name")
    fields_sample = ", ".join(field_names[:5]) if field_names else "*"
    count_col = field_names[0] if field_names else "id"

    query_patterns = [
        QueryPattern(
            intent="Show all",
            sql=f"SELECT {fields_sample} FROM {table} LIMIT 100",
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

    schema = EntitySchema(
        id=entity.id,
        entity_name=entity.name,
        display_name=entity.display_name or entity.name,
        description=entity.description,
        record_count=entity.record_count,
        fields=field_schemas,
    )
    return EntitySQLContext(entity_schema=schema, query_patterns=query_patterns)


def build_sql_context(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
    prompt_version: str | None = None,
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

    return SQLContext(
        base_system_prompt=base_system_prompt or None,
        resource_description=None,
        sql_expert_system_prompt=rendered_prompt,
        constraints=SQL_CONSTRAINTS,
        entity_contexts=[build_entity_context(e) for e in entities],
    )


def render_entity_schema_sections(ctx: SQLContext) -> list[str]:
    """Render the entity-schema tables + query patterns as prompt lines.

    Shared by the entity-tool prompt (:func:`format_sql_context`) and the
    ontology-tool prompt (``datafabric_ontology_prompt_builder``) so the
    per-entity rendering stays in one place while each tool owns its own
    top-level prompt assembly.
    """
    lines: list[str] = ["## All available Data Fabric Entities", ""]

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

    return lines


def format_sql_context(ctx: SQLContext) -> str:
    """Format a SQLContext as the entity-tool inner system prompt.

    This is the entity tool's prompt only — it carries no ontology content. The
    ontology tool assembles its own prompt (OWL + R2RML) in
    ``datafabric_ontology_prompt_builder`` so the two prompts stay independent.
    """
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

    lines.extend(render_entity_schema_sections(ctx))

    return "\n".join(lines)


def build(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
    prompt_version: str | None = None,
) -> str:
    """Build the entity-tool inner system prompt.

    Combines agent system prompt, the rendered SQL strategy prompt, the
    Calcite constraint deny-list, and entity schemas + query patterns. The
    ontology tool has its own builder (``datafabric_ontology_prompt_builder``).

    Args:
        entities: List of Entity objects with schema information.
        resource_description: Optional description of the resource/entity set;
            folded into the rendered prompt as domain guidance.
        base_system_prompt: Optional system prompt from the outer agent.
        prompt_version: Optional version key (e.g. ``"v0"``, ``"v1"``).
            Defaults to the registry's default.

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
    )
    return format_sql_context(ctx)
