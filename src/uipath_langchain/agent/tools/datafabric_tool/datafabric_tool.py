"""Data Fabric tool creation and resource detection.

This module provides:
1. A single generic ``query_datafabric`` tool
2. Entity schema fetching from the Data Fabric API
3. Helpers to extract entity identifiers from agent definitions

Schema building and formatting is in ``schema_context.py``.
"""

import logging
from typing import Any, Sequence

from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    BaseAgentResourceConfig,
    LowCodeAgentDefinition,
)
from uipath.platform.entities import Entity

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from .models import DataFabricQueryInput

logger = logging.getLogger(__name__)


# --- Schema Fetching ---


async def fetch_entity_schemas(entity_identifiers: list[str]) -> list[Entity]:
    """Fetch entity metadata from Data Fabric concurrently.

    Args:
        entity_identifiers: List of entity identifiers to fetch.

    Returns:
        List of Entity objects with full schema information.
    """
    import asyncio

    from uipath.platform import UiPath

    sdk = UiPath()

    async def _fetch_single(identifier: str) -> Entity | None:
        try:
            entity = await sdk.entities.retrieve_async(identifier)
            logger.info("Fetched schema for entity '%s'", entity.display_name)
            return entity
        except Exception:
            logger.warning("Failed to fetch entity '%s'", identifier, exc_info=True)
            return None

    results = await asyncio.gather(*[_fetch_single(eid) for eid in entity_identifiers])
    return [e for e in results if e is not None]


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


def get_datafabric_query_routing_context(
    resources: Sequence[BaseAgentResourceConfig],
) -> "QueryRoutingContext | None":
    """Build query routing context from DF entity set items.

    Maps each entity to its folder so the backend can resolve
    entities at folder level instead of tenant level.
    """
    from uipath.platform.entities import EntityRouting, QueryRoutingContext

    routings: list[EntityRouting] = []
    for context in _filter_datafabric_contexts(resources):
        for item in context.entity_set or []:
            routings.append(
                EntityRouting(
                    entity_name=item.name,
                    folder_id=item.folder_id,
                )
            )
    if not routings:
        return None
    return QueryRoutingContext(entity_routings=routings)


# --- Generic Tool Creation ---

_MAX_RECORDS_IN_RESPONSE = 50


_datafabric_tool_instance: BaseTool | None = None


def create_datafabric_query_tool(
    routing_context: "QueryRoutingContext | None" = None,
) -> BaseTool:
    """Create the ``query_datafabric`` tool.

    Returns a cached instance. The routing_context from the first call
    is captured by the closure and used for all subsequent queries.
    """
    global _datafabric_tool_instance
    if _datafabric_tool_instance is not None:
        return _datafabric_tool_instance

    async def _query_datafabric(sql_query: str) -> dict[str, Any]:
        from uipath.platform import UiPath
        import json as _json

        logger.debug(f"query_datafabric called with SQL: {sql_query}")

        # Debug: dump routing context and query
        with open("/tmp/df_query_debug.txt", "a") as _f:
            _f.write(f"SQL: {sql_query}\n")
            _f.write(f"routing_context: {routing_context.model_dump(by_alias=True) if routing_context else None}\n\n")

        sdk = UiPath()
        try:
            records = await sdk.entities.query_entity_records_async(
                sql_query=sql_query,
                routing_context=routing_context,
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
            with open("/tmp/df_query_debug.txt", "a") as _f:
                _f.write(f"Result: {total_count} records, truncated={truncated}\n\n")
            return result
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            with open("/tmp/df_query_debug.txt", "a") as _f:
                _f.write(f"Error: {e}\n\n")
            return {
                "records": [],
                "total_count": 0,
                "error": str(e),
                "sql_query": sql_query,
            }

    _datafabric_tool_instance = BaseUiPathStructuredTool(
        name="query_datafabric",
        description=(
            "Execute a SQL SELECT query against Data Fabric entities. "
            "Refer to the entity schemas in the system prompt for available tables and columns. "
            "Include LIMIT unless aggregating."
        ),
        args_schema=DataFabricQueryInput,
        coroutine=_query_datafabric,
        metadata={"tool_type": "datafabric_sql"},
    )
    return _datafabric_tool_instance
