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
from uipath.platform.entities import Entity, EntityRouting, QueryRoutingContext

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


def build_routing_context(
    contexts: list[AgentContextResourceConfig],
) -> QueryRoutingContext | None:
    """Build a QueryRoutingContext from Data Fabric context resources.

    Args:
        contexts: Data Fabric context resource configs.

    Returns:
        A QueryRoutingContext if any entity routings exist, otherwise None.
    """
    routings: list[EntityRouting] = []
    for context in contexts:
        if context.entity_set:
            for item in context.entity_set:
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


def create_datafabric_query_tool(
    routing_context: QueryRoutingContext | None = None,
) -> BaseTool:
    """Create the ``query_datafabric`` tool.

    Args:
        routing_context: Optional routing context for multi-folder entity queries.
    """

    async def _query_datafabric(sql_query: str) -> dict[str, Any]:
        from uipath.platform import UiPath

        logger.debug(f"query_datafabric called with SQL: {sql_query}")

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
        args_schema=DataFabricQueryInput,
        coroutine=_query_datafabric,
        metadata={"tool_type": "datafabric_sql"},
    )


def create_datafabric_tools(agent: LowCodeAgentDefinition) -> list[BaseTool]:
    """Create Data Fabric tools for an agent when Data Fabric context is configured.

    Args:
        agent: The agent definition.

    Returns:
        A list containing the generic Data Fabric query tool when at least one
        Data Fabric context resource is enabled, otherwise an empty list.
    """
    datafabric_contexts = get_datafabric_contexts(agent)
    if not datafabric_contexts:
        return []
    routing_context = build_routing_context(datafabric_contexts)
    return [create_datafabric_query_tool(routing_context=routing_context)]
