"""Schema context building and formatting for Data Fabric entities.

Converts raw Entity SDK objects into structured Pydantic models (SQLContext),
then formats them as text for system prompt injection.
"""

import logging
from functools import lru_cache
from pathlib import Path

from uipath.platform.entities import Entity

from .models import (
    EntitySchema,
    EntitySQLContext,
    FieldSchema,
    QueryPattern,
    SQLContext,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _load_sql_constraints() -> str:
    """Load SQL constraints from sql_constraints.txt."""
    constraints_path = _PROMPTS_DIR / "sql_constraints.txt"
    try:
        return constraints_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("SQL constraints file not found: %s", constraints_path)
        return ""


@lru_cache(maxsize=1)
def _load_sql_expert_system_prompt() -> str:
    """Load SQL generation strategy from sql_expert_system_prompt.txt."""
    prompt_path = _PROMPTS_DIR / "sql_expert_system_prompt.txt"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("System prompt file not found: %s", prompt_path)
        return ""


# --- Build ---


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
) -> SQLContext:
    """Build the full SQL context from entities, prompts, and constraints."""
    return SQLContext(
        base_system_prompt=base_system_prompt or None,
        resource_description=resource_description or None,
        sql_expert_system_prompt=_load_sql_expert_system_prompt(),
        constraints=_load_sql_constraints(),
        entity_contexts=[build_entity_context(e) for e in entities],
    )


# --- Format ---


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


# --- Public API ---


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
