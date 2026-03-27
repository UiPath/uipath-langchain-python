"""Data Fabric tool module for entity-based SQL queries."""

import logging
from typing import Sequence

from uipath.agent.models.agent import BaseAgentResourceConfig

from uipath_langchain.agent.react.init_context_registry import (
    register_init_context_provider,
)

from .datafabric_tool import (
    create_datafabric_query_tool,
    fetch_entity_schemas,
    get_datafabric_contexts,
    get_datafabric_entity_identifiers_from_resources,
)
from .schema_context import format_schemas_for_context

__all__ = [
    "create_datafabric_query_tool",
    "fetch_entity_schemas",
    "format_schemas_for_context",
    "get_datafabric_contexts",
    "get_datafabric_entity_identifiers_from_resources",
]

_logger = logging.getLogger(__name__)


# --- Init-time context self-registration ---


async def _datafabric_init_context_provider(
    resources: Sequence[BaseAgentResourceConfig],
) -> str | None:
    """Fetch and format DataFabric entity schemas for system prompt injection."""
    entity_identifiers = get_datafabric_entity_identifiers_from_resources(resources)
    if not entity_identifiers:
        return None

    _logger.info(
        "Fetching Data Fabric schemas for %d identifier(s)",
        len(entity_identifiers),
    )
    entities = await fetch_entity_schemas(entity_identifiers)
    return format_schemas_for_context(entities)


register_init_context_provider("datafabric", _datafabric_init_context_provider)
