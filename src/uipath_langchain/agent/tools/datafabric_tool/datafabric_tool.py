"""Data Fabric tool creation for entity-based queries.

This module provides:
1. A single generic ``query_datafabric`` tool (no per-entity knowledge at build time)
2. Schema fetching & formatting helpers consumed by the INIT node at runtime
3. Helpers to extract entity identifiers from agent definitions
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    BaseAgentResourceConfig,
    LowCodeAgentDefinition,
)
from uipath.platform.entities import Entity, FieldMetadata

from ..base_uipath_structured_tool import BaseUiPathStructuredTool

logger = logging.getLogger(__name__)

# --- Prompt and Constraints Loading ---

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _load_sql_constraints() -> str:
    """Load SQL constraints from sql_constraints.txt."""
    constraints_path = _PROMPTS_DIR / "sql_constraints.txt"
    try:
        return constraints_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(f"SQL constraints file not found: {constraints_path}")
        return ""


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    """Load SQL generation strategy from system_prompt.txt."""
    prompt_path = _PROMPTS_DIR / "system_prompt.txt"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(f"System prompt file not found: {prompt_path}")
        return ""


# --- Schema Fetching and Formatting ---


async def fetch_entity_schemas(entity_identifiers: list[str]) -> list[Entity]:
    """Fetch entity metadata from Data Fabric for the given entity identifiers.

    Args:
        entity_identifiers: List of entity identifiers to fetch.

    Returns:
        List of Entity objects with full schema information.
    """
    from uipath.platform import UiPath

    sdk = UiPath()
    entities: list[Entity] = []

    for entity_identifier in entity_identifiers:
        try:
            entity = await sdk.entities.retrieve_async(entity_identifier)
            entities.append(entity)
            logger.info(f"Fetched schema for entity '{entity.display_name}'")
        except Exception as e:
            logger.warning(f"Failed to fetch entity '{entity_identifier}': {e}")

    return entities


def format_field_type(field: FieldMetadata) -> str:
    """Format a field's type information for display."""
    type_name = field.sql_type.name if field.sql_type else "unknown"

    modifiers = []
    if field.is_primary_key:
        modifiers.append("PK")
    if field.is_foreign_key and field.reference_entity:
        ref_name = field.reference_entity.display_name or field.reference_entity.name
        modifiers.append(f"FK → {ref_name}")
    if field.is_required:
        modifiers.append("required")

    if modifiers:
        return f"{type_name}, {', '.join(modifiers)}"
    return type_name


def format_schemas_for_context(entities: list[Entity]) -> str:
    """Format entity schemas as markdown for injection into agent system prompt.

    The output is optimized for SQL query generation by the LLM.
    Includes: SQL strategy prompt, constraints, entity schemas, and query patterns.

    Args:
        entities: List of Entity objects with schema information.

    Returns:
        Markdown-formatted string describing entity schemas and SQL guidance.
    """
    if not entities:
        return ""

    lines: list[str] = []

    system_prompt = _load_system_prompt()
    if system_prompt:
        lines.append("## SQL Query Generation Guidelines")
        lines.append("")
        lines.append(system_prompt)
        lines.append("")

    sql_constraints = _load_sql_constraints()
    if sql_constraints:
        lines.append("## SQL Constraints")
        lines.append("")
        lines.append(sql_constraints)
        lines.append("")

    lines.append("## Available Data Fabric Entities")
    lines.append("")

    for entity in entities:
        sql_table = entity.name
        display_name = entity.display_name or entity.name
        lines.append(f"### Entity: {display_name} (SQL table: `{sql_table}`)")
        if entity.description:
            lines.append(f"_{entity.description}_")
        lines.append("")
        lines.append("| Field | Type |")
        lines.append("|-------|------|")

        field_names: list[str] = []
        numeric_field: str | None = None
        text_field: str | None = None

        for field in entity.fields or []:
            if field.is_hidden_field or field.is_system_field:
                continue
            field_name = field.name
            field_type = format_field_type(field)
            field_names.append(field_name)
            lines.append(f"| {field_name} | {field_type} |")

            sql_type = field.sql_type.name.lower() if field.sql_type else ""
            if not numeric_field and sql_type in (
                "int",
                "decimal",
                "float",
                "double",
                "bigint",
            ):
                numeric_field = field_name
            if not text_field and sql_type in (
                "varchar",
                "nvarchar",
                "text",
                "string",
                "ntext",
            ):
                text_field = field_name

        lines.append("")

        group_field = text_field or (field_names[0] if field_names else "Category")
        agg_field = numeric_field or (
            field_names[1] if len(field_names) > 1 else "Amount"
        )
        filter_field = text_field or (field_names[0] if field_names else "Name")
        fields_sample = ", ".join(field_names[:5]) if field_names else "*"

        lines.append(f"**Query Patterns for {sql_table}:**")
        lines.append("")
        lines.append("| User Intent | SQL Pattern |")
        lines.append("|-------------|-------------|")
        lines.append(
            f"| 'Show all' | `SELECT {fields_sample} FROM {sql_table} LIMIT 100` |"
        )
        lines.append(
            f"| 'Find by X' | `SELECT {fields_sample} FROM {sql_table} WHERE {filter_field} = 'value' LIMIT 100` |"
        )
        lines.append(
            f"| 'Top N by Y' | `SELECT {fields_sample} FROM {sql_table} ORDER BY {agg_field} DESC LIMIT N` |"
        )
        lines.append(
            f"| 'Count by X' | `SELECT {group_field}, COUNT(*) as count FROM {sql_table} GROUP BY {group_field}` |"
        )
        lines.append(
            f"| 'Top N segments' | `SELECT {group_field}, COUNT(*) as count FROM {sql_table} GROUP BY {group_field} ORDER BY count DESC LIMIT N` |"
        )
        lines.append(
            f"| 'Sum/Avg of Y' | `SELECT SUM({agg_field}) as total FROM {sql_table}` |"
        )
        lines.append("")

    return "\n".join(lines)


# --- Data Fabric Context Detection ---


def get_datafabric_contexts(
    agent: LowCodeAgentDefinition,
) -> list[AgentContextResourceConfig]:
    """Extract Data Fabric context resources from agent definition.

    Args:
        agent: The agent definition to search.

    Returns:
        List of context resources configured for Data Fabric retrieval mode.
    """
    return _filter_datafabric_contexts(agent.resources)


def _filter_datafabric_contexts(
    resources: Sequence[BaseAgentResourceConfig],
) -> list[AgentContextResourceConfig]:
    """Filter resources to only Data Fabric context configs."""
    return [
        resource
        for resource in resources
        if isinstance(resource, AgentContextResourceConfig)
        and resource.is_enabled
        and resource.is_datafabric
    ]


def get_datafabric_entity_identifiers_from_resources(
    resources: Sequence[BaseAgentResourceConfig],
) -> list[str]:
    """Extract Data Fabric entity identifiers from a sequence of resource configs.

    Args:
        resources: Resource configs (typically from ``agent_definition.resources``).

    Returns:
        Flat list of entity identifier strings across all Data Fabric contexts.
    """
    identifiers: list[str] = []
    for context in _filter_datafabric_contexts(resources):
        identifiers.extend(context.datafabric_entity_identifiers)
    return identifiers


# --- Generic Tool Creation ---

_MAX_RECORDS_IN_RESPONSE = 50


def create_datafabric_query_tool() -> BaseTool:
    """Create a single generic ``query_datafabric`` tool.

    The tool accepts an arbitrary SQL SELECT query and dispatches it to
    ``sdk.entities.query_entity_records_async()``.  Entity knowledge is
    *not* baked in — the LLM receives schema guidance via the system
    prompt (injected at INIT time) and constructs raw SQL.
    """

    async def _query_datafabric(sql_query: str) -> dict[str, Any]:
        from uipath.platform import UiPath

        logger.debug(f"query_datafabric called with SQL: {sql_query}")

        sdk = UiPath()
        try:
            records = await sdk.entities.query_entity_records_async(
                sql_query=sql_query,
            )
            total_count = len(records)
            truncated = total_count > _MAX_RECORDS_IN_RESPONSE
            returned_records = (
                records[:_MAX_RECORDS_IN_RESPONSE] if truncated else records
            )

            result: dict[str, Any] = {
                "records": returned_records,
                "total_count": total_count,
                "returned_count": len(returned_records),
                "sql_query": sql_query,
            }
            if truncated:
                result["truncated"] = True
                result["message"] = (
                    f"Showing {len(returned_records)} of {total_count} records. "
                    "Use more specific filters or LIMIT to narrow results."
                )
            return result
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return {
                "records": [],
                "total_count": 0,
                "error": str(e),
                "sql_query": sql_query,
            }

    return BaseUiPathStructuredTool(
        name="query_datafabric",
        description=(
            "Execute a SQL SELECT query against Data Fabric entities. "
            "Refer to the entity schemas in the system prompt for available tables and columns. "
            "Include LIMIT unless aggregating."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": (
                        "Complete SQL SELECT statement. "
                        "Use exact table and column names from the entity schemas in the system prompt."
                    ),
                },
            },
            "required": ["sql_query"],
        },
        coroutine=_query_datafabric,
        metadata={"tool_type": "datafabric_sql"},
    )


def create_datafabric_tools(
    agent: LowCodeAgentDefinition,
) -> list[BaseTool]:
    """Register the generic Data Fabric query tool if the agent has DF contexts.

    No fetching, no formatting, no schema — purely tool registration.
    Schema hydration happens at INIT time.

    Args:
        agent: The agent definition containing Data Fabric context resources.

    Returns:
        A list containing the single ``query_datafabric`` tool, or empty.
    """
    if not get_datafabric_entity_identifiers_from_resources(agent.resources):
        return []

    logger.info("Registering generic query_datafabric tool")
    return [create_datafabric_query_tool()]
