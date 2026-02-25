"""Data Fabric tool creation for entity-based queries.

This module provides functionality to:
1. Fetch and format entity schemas for agent context hydration
2. Create SQL-based query tools for the agent
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    LowCodeAgentDefinition,
)
from uipath.platform.entities import Entity, FieldMetadata

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from ..utils import sanitize_tool_name

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

    lines = []

    # Add SQL generation strategy from system_prompt.txt
    system_prompt = _load_system_prompt()
    if system_prompt:
        lines.append("## SQL Query Generation Guidelines")
        lines.append("")
        lines.append(system_prompt)
        lines.append("")

    # Add SQL constraints from sql_constraints.txt
    sql_constraints = _load_sql_constraints()
    if sql_constraints:
        lines.append("## SQL Constraints")
        lines.append("")
        lines.append(sql_constraints)
        lines.append("")

    lines.append("## Available Data Fabric Entities")
    lines.append("")

    for entity in entities:
        display_name = entity.display_name or entity.name
        lines.append(f"### Entity: {display_name}")
        if entity.description:
            lines.append(f"_{entity.description}_")
        lines.append("")
        lines.append("| Field | Type |")
        lines.append("|-------|------|")

        # Collect field info for query pattern examples
        field_names = []
        numeric_field = None
        text_field = None

        for field in entity.fields or []:
            if field.is_hidden_field or field.is_system_field:
                continue
            field_name = field.name
            field_type = format_field_type(field)
            field_names.append(field_name)
            display_label = f"{field_name} ({field.display_name})" if field.display_name and field.display_name != field_name else field_name
            lines.append(f"| {display_label} | {field_type} |")

            # Track field types for examples
            sql_type = field.sql_type.name.lower() if field.sql_type else ""
            if not numeric_field and sql_type in ("int", "decimal", "float", "double", "bigint"):
                numeric_field = field_name
            if not text_field and sql_type in ("varchar", "nvarchar", "text", "string", "ntext"):
                text_field = field_name

        lines.append("")

        # Add entity-specific query pattern examples
        group_field = text_field or (field_names[0] if field_names else "Category")
        agg_field = numeric_field or (field_names[1] if len(field_names) > 1 else "Amount")
        filter_field = text_field or (field_names[0] if field_names else "Name")
        fields_sample = ", ".join(field_names[:5]) if field_names else "*"

        sql_table_name = entity.name
        lines.append(f"**Query Patterns for {display_name} (SQL table: {sql_table_name}):**")
        lines.append("")
        lines.append("| User Intent | SQL Pattern |")
        lines.append("|-------------|-------------|")
        lines.append(f"| 'Show all {display_name.lower()}' | `SELECT {fields_sample} FROM {sql_table_name} LIMIT 100` |")
        lines.append(f"| 'Find by X' | `SELECT {fields_sample} FROM {sql_table_name} WHERE {filter_field} = 'value' LIMIT 100` |")
        lines.append(f"| 'Top N by Y' | `SELECT {fields_sample} FROM {sql_table_name} ORDER BY {agg_field} DESC LIMIT N` |")
        lines.append(f"| 'Count by X' | `SELECT {group_field}, COUNT({field_names[0] if field_names else 'id'}) as count FROM {sql_table_name} GROUP BY {group_field}` |")
        lines.append(f"| 'Top N segments' | `SELECT {group_field}, COUNT({field_names[0] if field_names else 'id'}) as count FROM {sql_table_name} GROUP BY {group_field} ORDER BY count DESC LIMIT N` |")
        lines.append(f"| 'Sum/Avg of Y' | `SELECT SUM({agg_field}) as total FROM {sql_table_name}` |")
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
    datafabric_contexts: list[AgentContextResourceConfig] = []

    for resource in agent.resources:
        if not isinstance(resource, AgentContextResourceConfig):
            continue
        if not resource.is_enabled:
            continue
        if resource.settings.retrieval_mode.lower() == "datafabric":
            datafabric_contexts.append(resource)

    return datafabric_contexts


# --- Tool Creation ---


class QueryEntityInput(BaseModel):
    """Input schema for the query_entity tool."""

    entity_identifier: str = Field(
        ..., description="The entity identifier to query"
    )
    sql_where: str = Field(
        default="",
        description="SQL WHERE clause to filter records (without the WHERE keyword). "
        "Example: 'Status = \"Active\" AND Amount > 100'",
    )
    limit: int = Field(
        default=1000, description="Maximum number of records to return"
    )


class QueryEntityOutput(BaseModel):
    """Output schema for the query_entity tool."""

    records: list[dict[str, Any]] = Field(
        ..., description="List of entity records matching the query"
    )
    total_count: int = Field(..., description="Total number of matching records")


async def create_datafabric_tools(
    agent: LowCodeAgentDefinition,
) -> tuple[list[BaseTool], str]:
    """Create Data Fabric tools and schema context from agent definition.

    This function:
    1. Finds all Data Fabric context resources in the agent
    2. Fetches entity schemas for context hydration
    3. Returns tools and schema context string

    Args:
        agent: The agent definition containing Data Fabric context resources.

    Returns:
        Tuple of (tools, schema_context) where:
        - tools: List of BaseTool instances for querying entities
        - schema_context: Formatted schema string to inject into system prompt
    """
    tools: list[BaseTool] = []  
    all_entities: list[Entity] = []

    datafabric_contexts = get_datafabric_contexts(agent)

    if not datafabric_contexts:
        return tools, ""

    logger.info(f"Found {len(datafabric_contexts)} Data Fabric context resource(s)")

    for context in datafabric_contexts:
        entity_identifiers = context.settings.entity_identifiers or []

        if not entity_identifiers:
            logger.warning(
                f"Data Fabric context '{context.name}' has no entity_identifiers configured"
            )
            continue

        # Fetch entity schemas
        entities = await fetch_entity_schemas(entity_identifiers)
        all_entities.extend(entities)

        context_tools = _create_sdk_based_tools(context, entities)
        tools.extend(context_tools)

        logger.info(
            f"Created {len(context_tools)} tools for Data Fabric context '{context.name}'"
        )

    # Format all entity schemas for context injection
    schema_context = format_schemas_for_context(all_entities)

    return tools, schema_context


def _create_sdk_based_tools(
    context: AgentContextResourceConfig,
    entities: list[Entity],
) -> list[BaseTool]:
    """Create SDK-based tools for querying entities using SQL.

    Each tool accepts a full SQL query and executes it via the SDK's
    query_entity_records_async method.
    """
    tools: list[BaseTool] = []
    MAX_RECORDS_IN_RESPONSE = 50  # Limit records to prevent context overflow

    for entity in entities:
        tool_name = sanitize_tool_name(f"query_{entity.name}")

        # Create a closure to capture the entity name
        entity_display_name = entity.display_name or entity.name

        async def query_fn(
            sql_query: str,
            _entity_name: str = entity_display_name,
            _max_records: int = MAX_RECORDS_IN_RESPONSE,
        ) -> dict[str, Any]:
            """Execute a SQL query against the Data Fabric entity."""
            from uipath.platform import UiPath

            print(f"[DEBUG] query_fn called for entity '{_entity_name}' with SQL: {sql_query}")
            logger.info(f"Executing SQL query for entity '{_entity_name}': {sql_query}")

            sdk = UiPath()
            try:
                records = await sdk.entities.query_entity_records_async(
                    sql_query=sql_query,
                )
                total_count = len(records)
                truncated = total_count > _max_records
                returned_records = records[:_max_records] if truncated else records
                
                print(f"[DEBUG] Retrieved {total_count} records, returning {len(returned_records)} for entity '{_entity_name}'")
                
                result = {
                    "records": returned_records,
                    "total_count": total_count,
                    "returned_count": len(returned_records),
                    "entity": _entity_name,
                    "sql_query": sql_query,
                }
                if truncated:
                    result["truncated"] = True
                    result["message"] = f"Showing {len(returned_records)} of {total_count} records. Use more specific filters or LIMIT to narrow results."

                # Emit result to file for eval pipeline (activated via env var)
                result_file = os.environ.get("DATAFABRIC_RESULT_FILE")
                if result_file:
                    with open(result_file, "w") as f:
                        json.dump(result, f)

                return result
            except Exception as e:
                logger.error(f"SQL query failed for entity '{_entity_name}': {e}")
                return {
                    "records": [],
                    "total_count": 0,
                    "error": str(e),
                    "sql_query": sql_query,
                }

        entity_description = entity.description or f"Query {entity_display_name} records"

        # Extract actual field names from entity schema (exclude system/hidden fields)
        field_names = [
            f.display_name or f.name
            for f in (entity.fields or [])
            if not f.is_hidden_field and not f.is_system_field
        ]
        fields_str = ", ".join(field_names[:10])  # Limit to first 10 fields
        if len(field_names) > 10:
            fields_str += f", ... ({len(field_names)} total)"

        tools.append(
            BaseUiPathStructuredTool(
                name=tool_name,
                description=(
                    f"{context.description}. {entity_description}. "
                    f"Available fields: {fields_str}. "
                    f"Use SQL patterns from system prompt based on user intent."
                ),
                args_schema={
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": (
                                f"Complete SQL SELECT statement for {entity_display_name}. "
                                f"Use exact column names from schema. Include LIMIT unless aggregating."
                            ),
                        },
                    },
                    "required": ["sql_query"],
                },
                coroutine=query_fn,
                metadata={
                    "tool_type": "datafabric_sql",
                    "display_name": f"Query {entity_display_name}",
                    "entity_name": entity_display_name,
                },
            )
        )

    return tools
