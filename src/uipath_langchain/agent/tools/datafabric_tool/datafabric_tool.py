"""Data Fabric tool creation and resource detection.

This module provides an agentic ``query_datafabric`` tool with an inner
LLM sub-graph.

The tool accepts natural language queries, runs an inner LangGraph
sub-graph for SQL generation + execution + self-correction, and
returns a natural language answer.

Prompt building is in ``datafabric_prompt_builder.py``.
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
from uipath.platform.entities import DataFabricEntityItem

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from .models import DataFabricQueryInput

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = "base_system_prompt"


class DataFabricTextQueryHandler:
    """Manages lazy initialization and invocation of the Data Fabric sub-graph.

    On first call, resolves entity schemas and routing via the platform
    layer and compiles the inner LangGraph sub-graph. Subsequent calls
    reuse the cached graph.
    """

    def __init__(
        self,
        entity_set: list[DataFabricEntityItem],
        llm: BaseChatModel,
        resource_description: str = "",
        base_system_prompt: str = "",
    ) -> None:
        self._entity_set = entity_set
        self._llm = llm
        self._resource_description = resource_description
        self._base_system_prompt = base_system_prompt
        self._compiled: CompiledStateGraph[Any] | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_datafabric_graph(self) -> CompiledStateGraph[Any]:
        """Lazy-init: resolve entities + build sub-graph on first call.

        Uses asyncio.Lock because the outer agent supports parallel
        tool calls — two concurrent invocations could race on first call.
        """
        if self._compiled is not None:
            return self._compiled

        async with self._init_lock:
            if self._compiled is not None:
                return self._compiled

            from uipath.platform import UiPath

            from .datafabric_subgraph import DataFabricGraph

            sdk = UiPath()
            resolution = await sdk.entities.resolve_entity_set_async(self._entity_set)
            if not resolution.entities:
                raise ValueError(
                    "No Data Fabric entity schemas could be fetched. "
                    "Check entity identifiers and permissions."
                )
            self._compiled = DataFabricGraph.create(
                llm=self._llm,
                entities=resolution.entities,
                entities_service=resolution.entities_service,
                resource_description=self._resource_description,
                base_system_prompt=self._base_system_prompt,
            )
            return self._compiled

    async def __call__(self, user_query: str) -> str:
        logger.debug("query_datafabric called with: %s", user_query)

        compiled_graph = await self._ensure_datafabric_graph()
        result_state = await compiled_graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}
        )
        for msg in reversed(result_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)

        return "Unable to generate an answer from the available data."


def create_datafabric_query_tool(
    resource: AgentContextResourceConfig,
    llm: BaseChatModel,
    tool_name: str = "query_datafabric",
    agent_config: dict[str, str] | None = None,
) -> BaseTool:
    """Create the ``query_datafabric`` agentic tool.

    Args:
        resource: The Data Fabric context resource configuration.
        llm: The language model for the inner SQL generation loop.
        tool_name: Sanitized tool name from the resource.
        agent_config: Optional dict with agent-level config.
            Key ``base_system_prompt`` carries the outer agent's system prompt.
    """
    config = agent_config or {}
    entity_set = [
        DataFabricEntityItem.model_validate(item.model_dump(by_alias=True))
        for item in (resource.entity_set or [])
    ]
    handler = DataFabricTextQueryHandler(
        entity_set=entity_set,
        llm=llm,
        resource_description=resource.description or "",
        base_system_prompt=config.get(BASE_SYSTEM_PROMPT, ""),
    )
    return BaseUiPathStructuredTool(
        name=tool_name,
        description=(
            "Query Data Fabric entities using natural language. "
            "Describe what data you need and the tool will translate it to SQL, "
            "execute the query, and return a natural language answer."
        ),
        args_schema=DataFabricQueryInput,
        coroutine=handler,
        metadata={"tool_type": "datafabric_sql"},
    )
