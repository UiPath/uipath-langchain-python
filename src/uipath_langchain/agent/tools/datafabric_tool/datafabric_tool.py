"""Data Fabric tool creation and resource detection.

This module provides:
1. An agentic ``query_datafabric`` tool with inner LLM sub-graph
2. Entity schema fetching from the Data Fabric API
3. Helpers to extract entity identifiers from agent definitions

The tool accepts natural language queries, runs an inner LangGraph
sub-graph for SQL generation + execution + self-correction, and
returns a natural language answer.

Schema building and formatting is in ``schema_context.py``.
Sub-graph definition is in ``datafabric_subgraph.py``.
"""

import asyncio
import logging
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
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
    """Extract Data Fabric context resources from agent definition."""
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
    """Extract Data Fabric entity identifiers from a sequence of resource configs."""
    identifiers: list[str] = []
    for context in _filter_datafabric_contexts(resources):
        identifiers.extend(context.datafabric_entity_identifiers)
    return identifiers


# --- Routing Context ---


def _build_routing_context(
    resource: AgentContextResourceConfig,
) -> Any:
    """Build query routing context from entity set items.

    Maps each entity to its folder so the backend resolves
    entities at folder level instead of tenant level.

    Returns None if routing models are unavailable or no entity set exists.
    """
    try:
        from uipath.platform.entities import EntityRouting, QueryRoutingOverrideContext
    except ImportError:
        return None

    if not resource.entity_set:
        return None

    routings = [
        EntityRouting(
            entity_name=item.name,
            folder_id=item.folder_id,
        )
        for item in resource.entity_set
    ]
    if not routings:
        return None
    return QueryRoutingOverrideContext(entity_routings=routings)


# --- Agentic Tool Creation ---


def create_datafabric_query_tool(
    resource: AgentContextResourceConfig,
    llm: BaseChatModel,
) -> BaseTool:
    """Create the ``query_datafabric`` agentic tool.

    The tool accepts natural language queries, runs an inner LangGraph
    sub-graph for SQL generation + execution + self-correction, and
    returns a natural language answer.

    Schemas are fetched lazily on first invocation and cached.

    Args:
        resource: The Data Fabric context resource configuration.
        llm: The language model for the inner SQL generation loop.
    """
    entity_identifiers = resource.datafabric_entity_identifiers
    routing_context = _build_routing_context(resource)

    _cache: dict[str, Any] = {}
    _init_lock = asyncio.Lock()

    async def _ensure_subgraph() -> Any:
        """Lazy-init: fetch schemas + build sub-graph on first call."""
        if "compiled" not in _cache:
            async with _init_lock:
                if "compiled" not in _cache:
                    from .datafabric_subgraph import create_datafabric_subgraph

                    entities = await fetch_entity_schemas(entity_identifiers)
                    if not entities:
                        raise ValueError(
                            "No Data Fabric entity schemas could be fetched. "
                            "Check entity identifiers and permissions."
                        )
                    _cache["compiled"] = create_datafabric_subgraph(
                        llm=llm,
                        entities=entities,
                        routing_context=routing_context,
                    )
        return _cache["compiled"]

    async def _query_datafabric(user_query: str) -> str:
        logger.debug("query_datafabric called with: %s", user_query)

        compiled_graph = await _ensure_subgraph()

        from .datafabric_subgraph import DataFabricSubgraphState

        initial_state = DataFabricSubgraphState(
            messages=[HumanMessage(content=user_query)],
        )

        result_state = await compiled_graph.ainvoke(initial_state)

        # Debug: dump full agent output
        import json as _json

        with open("/tmp/df_agent_debug.txt", "w") as _f:
            _f.write(f"User query: {user_query}\n\n")
            _f.write(f"Routing context: {routing_context}\n\n")
            for i, msg in enumerate(result_state["messages"]):
                _f.write(f"--- Message {i} ({type(msg).__name__}) ---\n")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        _f.write(f"Tool call: {tc.get('name')} args={tc.get('args')}\n")
                if hasattr(msg, "content") and msg.content:
                    _f.write(f"Content: {msg.content}\n")
                _f.write("\n")

        for msg in reversed(result_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return "Unable to generate an answer from the available data."

    return BaseUiPathStructuredTool(
        name="query_datafabric",
        description=(
            "Query Data Fabric entities using natural language. "
            "Describe what data you need and the tool will translate it to SQL, "
            "execute the query, and return a natural language answer."
        ),
        args_schema=DataFabricQueryInput,
        coroutine=_query_datafabric,
        metadata={"tool_type": "datafabric_sql"},
    )
