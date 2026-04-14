"""Schema context building and formatting for Data Fabric entities.

Converts raw Entity SDK objects into structured Pydantic models (SQLContext),
then formats them as text for system prompt injection.

Note: This module will go through refinements as we better understand
the tool's performance characteristics and scoring in production.
"""

import logging

from uipath.platform.entities import Entity

from .datafabric_prompts import SQL_GENERATION_GUIDE
from .models import (
    EntitySchema,
    EntitySQLContext,
    FieldSchema,
    QueryPattern,
    SQLContext,
)
from .optimizer.export import load_optimized_prompts

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
            is_primary_key=field.is_primary_key,
            is_foreign_key=field.is_foreign_key,
            is_external_field=field.is_external_field,
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
) -> SQLContext:
    """Build the full SQL context from entities, prompts, and constraints.

    If an ``optimized_prompts.json`` file exists (produced by the DSPy
    optimizer), its instruction and few-shot examples override the
    default SQL_GENERATION_GUIDE.  Schema enrichment is always applied.
    """
    optimized = load_optimized_prompts()
    if optimized:
        instruction = optimized.get("optimized_instruction", SQL_GENERATION_GUIDE)
        # Append few-shot examples to the instruction block
        few_shots = optimized.get("few_shot_examples", [])
        if few_shots:
            parts = [instruction, "", "FEW-SHOT EXAMPLES:"]
            for shot in few_shots:
                parts.append(f"  Q: {shot.get('question', '')}")
                parts.append(f"  SQL: {shot.get('sql', '')}")
                parts.append("")
            instruction = "\n".join(parts)
        logger.info(
            "Using DSPy-optimized prompts (accuracy: %.1f%%)",
            optimized.get("optimization_metadata", {}).get("val_accuracy", 0) * 100,
        )
    else:
        instruction = SQL_GENERATION_GUIDE

    return SQLContext(
        base_system_prompt=base_system_prompt or None,
        resource_description=resource_description or None,
        sql_expert_system_prompt=instruction,
        constraints=None,
        entity_contexts=[build_entity_context(e) for e in entities],
    )


def format_sql_context(ctx: SQLContext) -> str:
    """Format a SQLContext as text for system prompt injection.

    Ordering is optimized for LLM attention: entity schemas first (primacy),
    then query patterns, SQL rules, and finally agent instructions.
    """
    lines: list[str] = []

    # 1. Entity schemas first (most critical — primacy effect)
    lines.append("## Available Data Fabric Entities")
    lines.append("")

    if ctx.resource_description:
        lines.append(f"_{ctx.resource_description}_")
        lines.append("")

    for entity_ctx in ctx.entity_contexts:
        entity = entity_ctx.entity_schema
        header = (
            f"### Entity: {entity.display_name} (SQL table: `{entity.entity_name}`)"
        )
        lines.append(header)
        meta_parts: list[str] = []
        if entity.description:
            meta_parts.append(entity.description)
        if entity.record_count is not None:
            meta_parts.append(f"{entity.record_count:,} records")
        if meta_parts:
            lines.append(f"_{' | '.join(meta_parts)}_")
        lines.append("")
        lines.append("| Field | Type | Key | Description |")
        lines.append("|-------|------|-----|-------------|")

        for field in entity.fields:
            lines.append(
                f"| {field.name} | {field.display_type} "
                f"| {field.key_marker} | {field.short_description} |"
            )

        lines.append("")

        # Query patterns
        lines.append(f"**Query Patterns for {entity.entity_name}:**")
        lines.append("")
        lines.append("| User Intent | SQL Pattern |")
        lines.append("|-------------|-------------|")
        for p in entity_ctx.query_patterns:
            lines.append(f"| '{p.intent}' | `{p.sql}` |")
        lines.append("")

    # 2. SQL generation guide (consolidated rules + CoT)
    if ctx.sql_expert_system_prompt:
        lines.append("## SQL Query Generation Guide")
        lines.append("")
        lines.append(ctx.sql_expert_system_prompt)
        lines.append("")

    # 3. Agent instructions last (least critical)
    if ctx.base_system_prompt:
        lines.append("## Agent Instructions")
        lines.append("")
        lines.append(ctx.base_system_prompt)
        lines.append("")

    return "\n".join(lines)


def build(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
) -> str:
    """Build the full SQL prompt text for the inner sub-graph LLM.

    Combines agent system prompt, resource description, SQL guidelines,
    constraints, entity schemas, and query patterns into a single prompt string.

    Args:
        entities: List of Entity objects with schema information.
        resource_description: Optional description of the resource/entity set.
        base_system_prompt: Optional system prompt from the outer agent.

    Returns:
        Formatted prompt string for the inner LLM system message.
    """
    if not entities:
        return ""

    ctx = build_sql_context(entities, resource_description, base_system_prompt)
    return format_sql_context(ctx)
