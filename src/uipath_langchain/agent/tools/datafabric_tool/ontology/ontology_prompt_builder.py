"""Inner system-prompt builder for the Data Fabric **ontology** tool.

Separate from the entity tool's ``datafabric_prompt_builder`` so the two prompts
can evolve independently. This builder assembles an ontology-grounded prompt:
the OWL (authoritative semantic schema) and the R2RML (ontology→table/column
mapping) come first, then the SQL strategy/constraints and the entity schema
tables.

It reuses only ``build_sql_context`` (entity-context + strategy prompt
construction) from ``datafabric_prompt_builder``; the top-level layout, the
ontology-specific framing, and the entity-schema rendering are all owned here so
the shared entity-tool builder is not modified by this feature.
"""

from uipath.platform.entities import Entity

from ..datafabric_prompt_builder import build_sql_context
from ..models import SQLContext


def _render_entity_schema_sections(ctx: SQLContext) -> list[str]:
    """Render the entity-schema tables + query patterns as prompt lines."""
    lines: list[str] = ["## All available Data Fabric Entities", ""]

    for entity_ctx in ctx.entity_contexts:
        entity = entity_ctx.entity_schema
        lines.append(
            f"### Entity: {entity.display_name} (SQL table: `{entity.entity_name}`)"
        )
        if entity.description:
            lines.append(f"_{entity.description}_")
        lines.append("")
        lines.append("| Field | Type | Description |")
        lines.append("|-------|------|-------------|")
        for field in entity.fields:
            desc = (field.description or "").replace("|", r"\|").replace("\n", " ")
            lines.append(f"| {field.name} | {field.display_type} | {desc} |")

        lines.append("")

        lines.append(f"**Query Patterns for {entity.entity_name}:**")
        lines.append("")
        lines.append("| User Intent | SQL Pattern |")
        lines.append("|-------------|-------------|")
        for p in entity_ctx.query_patterns:
            lines.append(f"| '{p.intent}' | `{p.sql}` |")
        lines.append("")

    return lines


def format_ontology_context(
    ctx: SQLContext,
    ontology_text: str = "",
    r2rml_text: str = "",
) -> str:
    """Format a SQLContext + ontology artifacts as the ontology-tool prompt."""
    lines: list[str] = []

    if ctx.base_system_prompt:
        lines.append("## Agent Instructions")
        lines.append("")
        lines.append(ctx.base_system_prompt)
        lines.append("")

    if ontology_text:
        lines.append(
            "## Available Ontology (authoritative semantic schema)\n\n"
            "The ontology below is the authoritative source for the exact column "
            "names, value formats (date formats, codes, zero-padding), allowed "
            "values, and the relationships between entities — richer and more "
            "reliable than the field list further down, which omits value formats "
            "and semantics. Base your column names, filter values, and joins on "
            "it; when it and the entity tables disagree, the ontology wins.\n\n"
            f"{ontology_text}"
        )
        lines.append("")

    if r2rml_text:
        lines.append(
            "## Ontology→Table Mapping (R2RML)\n\n"
            "The R2RML below maps the ontology to the physical tables and columns "
            "you must use in SQL: `rr:tableName` is the SQL table, `rr:column` is "
            "the SQL column for each ontology predicate, and `rr:joinCondition` "
            "(child/parent) gives the exact columns to join on. Use it to turn "
            "ontology terms into precise SQL identifiers and joins.\n\n"
            f"{r2rml_text}"
        )
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
        lines.append("## Ontology description")
        lines.append("")
        lines.append(ctx.resource_description)
        lines.append("")

    lines.extend(_render_entity_schema_sections(ctx))

    return "\n".join(lines)


def build(
    entities: list[Entity],
    resource_description: str = "",
    base_system_prompt: str = "",
    ontology_text: str = "",
    r2rml_text: str = "",
    prompt_version: str | None = None,
) -> str:
    """Build the ontology-tool inner system prompt.

    Args:
        entities: Resolved Data Fabric entities (from the R2RML allow-list).
        resource_description: Optional description of the ontology context.
        base_system_prompt: Optional system prompt from the outer agent.
        ontology_text: The fetched ontology OWL content (authoritative schema).
        r2rml_text: The fetched ontology R2RML mapping (ontology→table/column).
        prompt_version: Optional SQL-strategy prompt version key.

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
    return format_ontology_context(
        ctx, ontology_text=ontology_text, r2rml_text=r2rml_text
    )
