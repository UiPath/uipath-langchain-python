"""Data Fabric tool creation and resource detection.

This module provides:
1. An agentic ``query_datafabric`` tool with inner LLM sub-graph
2. Entity schema fetching from the Data Fabric API

The tool accepts natural language queries, runs an inner LangGraph
sub-graph for SQL generation + execution + self-correction, and
returns a natural language answer.

SQL prompt building is in ``sql_prompt_builder.py``.
Sub-graph definition is in ``datafabric_subgraph.py``.
"""

import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.platform.entities import Entity, EntityRouting, QueryRoutingOverrideContext

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from .models import DataFabricQueryInput

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = "base_system_prompt"

class NLQueryHandler:
    """Manages lazy initialization and invocation of the Data Fabric sub-graph.

    On first call, fetches entity schemas from the DF API and compiles
    the inner LangGraph sub-graph. Subsequent calls reuse the cached graph.
    """

    def __init__(
        self,
        entity_identifiers: list[str],
        routing_context: QueryRoutingOverrideContext,
        llm: BaseChatModel,
        resource_description: str = "",
        base_system_prompt: str = "",
    ) -> None:
        self._entity_identifiers = entity_identifiers
        self._routing_context = routing_context
        self._llm = llm
        self._resource_description = resource_description
        self._base_system_prompt = base_system_prompt
        self._compiled: CompiledStateGraph | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_datafabric_graph(self) -> CompiledStateGraph:
        """Lazy-init: fetch schemas + build sub-graph on first call.

        Uses asyncio.Lock because the outer agent supports parallel
        tool calls — two concurrent invocations could race on first call.
        """
        if self._compiled is not None:
            return self._compiled

        async with self._init_lock:
            if self._compiled is not None:
                return self._compiled

            from .datafabric_subgraph import DataFabricGraph

            entities = await fetch_entity_schemas(self._entity_identifiers)
            if not entities:
                raise ValueError(
                    "No Data Fabric entity schemas could be fetched. "
                    "Check entity identifiers and permissions."
                )
            datafabric_graph = DataFabricGraph(
                llm=self._llm,
                entities=entities,
                routing_context=self._routing_context,
                resource_description=self._resource_description,
                base_system_prompt=self._base_system_prompt,
            )
            self._compiled = datafabric_graph.compile()
            return self._compiled

    async def __call__(self, user_query: str) -> str:
        logger.debug("query_datafabric called with: %s", user_query)

        compiled_graph = await self._ensure_datafabric_graph()
        result_state = await compiled_graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}
        )
        for msg in reversed(result_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return "Unable to generate an answer from the available data."


async def _fetch_single_entity(sdk: Any, identifier: str) -> Entity | None:
    """Fetch a single entity by identifier, returning None on failure."""
    try:
        entity = await sdk.entities.retrieve_async(identifier)
        logger.info("Fetched schema for entity '%s'", entity.display_name)
        return entity
    except Exception:
        logger.warning("Failed to fetch entity '%s'", identifier, exc_info=True)
        return None


async def fetch_entity_schemas(entity_identifiers: list[str]) -> list[Entity]:
    """Fetch entity metadata from Data Fabric concurrently."""
    from uipath.platform import UiPath

    sdk = UiPath()
    results = await asyncio.gather(
        *[_fetch_single_entity(sdk, eid) for eid in entity_identifiers]
    )
    return [e for e in results if e is not None]


def _build_routing_context(
    resource: AgentContextResourceConfig,
) -> QueryRoutingOverrideContext:
    """Build query routing context from entity set items.

    Maps each entity to its folder so the backend resolves
    entities at folder level instead of tenant level.
    """
    return QueryRoutingOverrideContext(
        entity_routings=[
            EntityRouting(entity_name=item.name, folder_id=item.folder_id)
            for item in resource.entity_set
        ]
    )


# --- Tool Creation ---


def create_datafabric_query_tool(
    resource: AgentContextResourceConfig,
    llm: BaseChatModel,
    agent_config: dict[str, str] | None = None,
) -> BaseTool:
    """Create the ``query_datafabric`` agentic tool.

    Args:
        resource: The Data Fabric context resource configuration.
        llm: The language model for the inner SQL generation loop.
        agent_config: Optional dict with agent-level config.
            Key ``base_system_prompt`` carries the outer agent's system prompt.
    """
    config = agent_config or {}
    handler = NLQueryHandler(
        entity_identifiers=resource.datafabric_entity_identifiers,
        routing_context=_build_routing_context(resource),
        llm=llm,
        resource_description=resource.description or "",
        base_system_prompt=config.get(BASE_SYSTEM_PROMPT, ""),
    )
    return BaseUiPathStructuredTool(
        name="query_datafabric",
        description=(
            "Query Data Fabric entities using natural language. "
            "Describe what data you need and the tool will translate it to SQL, "
            "execute the query, and return a natural language answer."
        ),
        args_schema=DataFabricQueryInput,
        coroutine=handler,
        metadata={"tool_type": "datafabric_sql"},
    )
