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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.platform.entities import DataFabricEntityItem

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from .models import DataFabricQueryInput

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = "base_system_prompt"

# Flag routing Data Fabric entity-metadata resolution to the V3 API.
ENTITY_V3_API_FF = "EnableEntityV3API"


@dataclass(frozen=True, slots=True)
class _DataFabricToolConfig:
    """Framework-neutral configuration for a Data Fabric query tool.

    Low-code contexts and coded-agent calls are normalized into this model before
    the LangChain tool and its lazy query handler are created.
    """

    name: str
    description: str
    entities: tuple[DataFabricEntityItem, ...]
    resource_description: str = ""
    base_system_prompt: str = ""


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

            from uipath.core.feature_flags import FeatureFlags
            from uipath.platform import UiPath

            from .datafabric_subgraph import DataFabricGraph

            sdk = UiPath()
            # Flag on: resolve via the V3 API method; off: the default method.
            if FeatureFlags.is_flag_enabled(ENTITY_V3_API_FF, default=False):
                resolution = await sdk.entities.resolve_entity_set_v3_async(
                    self._entity_set
                )
            else:
                resolution = await sdk.entities.resolve_entity_set_async(
                    self._entity_set
                )
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
        messages = result_state["messages"]
        last_message = messages[-1] if messages else None

        # On the happy path the sub-graph short-circuits at END after a
        # successful execute_sql call, so the terminal state contains one or
        # more ToolMessages. Collapse the trailing batch into one synthetic
        # message so the outer agent can reason over the full result set.
        if isinstance(last_message, ToolMessage):
            trailing_tool_messages: list[ToolMessage] = []
            for msg in reversed(messages):
                if not isinstance(msg, ToolMessage):
                    break
                trailing_tool_messages.append(msg)
            return self._format_terminal_tool_messages(
                list(reversed(trailing_tool_messages))
            )

        # On errors / max-iterations the terminal message is an AIMessage
        # carrying the natural-language explanation.
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)

        return "Unable to generate an answer from the available data."

    @staticmethod
    def _format_terminal_tool_messages(tool_messages: list[ToolMessage]) -> str:
        """Build one returned message from the terminal ToolMessage batch."""
        non_empty_contents = [
            str(msg.content) for msg in tool_messages if getattr(msg, "content", None)
        ]
        if not non_empty_contents:
            return "Unable to generate an answer from the available data."
        if len(non_empty_contents) == 1:
            return non_empty_contents[0]

        rendered_results = [
            f"Result {index}:\n{content}"
            for index, content in enumerate(non_empty_contents, start=1)
        ]
        return (
            "Multiple SQL queries executed successfully. "
            "Use all of the following results to answer the user's question.\n\n"
            + "\n\n".join(rendered_results)
        )


def _normalize_entities(
    entities: Sequence[DataFabricEntityItem],
) -> tuple[DataFabricEntityItem, ...]:
    """Copy entity references so caller mutations cannot change a built tool."""
    return tuple(
        DataFabricEntityItem.model_validate(entity.model_dump(by_alias=True))
        for entity in entities
    )


def _default_tool_description(entities: Sequence[DataFabricEntityItem]) -> str:
    entity_lines = []
    for entity in entities:
        line = f"- {entity.name}"
        if entity.description:
            line += f": {entity.description}"
        entity_lines.append(line)
    entity_summary = "\n".join(entity_lines)
    return (
        "Query the following Data Fabric entities using natural language:\n"
        f"{entity_summary}\n"
        "Describe what data you need and the tool will translate it to SQL, "
        "execute the query, and return a natural language answer."
    )


def _build_datafabric_tool(
    config: _DataFabricToolConfig,
    llm: BaseChatModel,
) -> BaseTool:
    """Build the shared LangChain tool used by coded and low-code agents."""
    handler = DataFabricTextQueryHandler(
        entity_set=list(config.entities),
        llm=llm,
        resource_description=config.resource_description,
        base_system_prompt=config.base_system_prompt,
    )
    return BaseUiPathStructuredTool(
        name=config.name,
        description=config.description,
        args_schema=DataFabricQueryInput,
        coroutine=handler.__call__,
        metadata={"tool_type": "datafabric_sql"},
    )


def create_datafabric_query_tool(
    resource: AgentContextResourceConfig,
    llm: BaseChatModel,
    tool_name: str = "query_datafabric",
    agent_config: dict[str, str] | None = None,
) -> BaseTool:
    """Create the low-code Data Fabric query tool from a context resource.

    Entity schemas and runtime binding overrides are resolved lazily on the first
    invocation. Keep the resulting tool scoped to one agent execution so its
    cached schema and routing cannot cross execution contexts.

    Args:
        resource: Low-code Data Fabric context resource.
        llm: Language model for the inner SQL generation loop.
        tool_name: LangChain tool name exposed to the agent.
        agent_config: Optional agent-level configuration. Key
            ``base_system_prompt`` carries the outer agent's system prompt.
    """
    config = agent_config or {}
    entity_set = _normalize_entities(resource.entity_set or [])

    return _build_datafabric_tool(
        _DataFabricToolConfig(
            name=tool_name,
            description=_default_tool_description(entity_set),
            entities=entity_set,
            resource_description=resource.description or "",
            base_system_prompt=config.get(BASE_SYSTEM_PROMPT, ""),
        ),
        llm,
    )


def create_datafabric_tool(
    *,
    llm: BaseChatModel,
    name: str,
    description: str,
    entities: Sequence[DataFabricEntityItem],
    base_system_prompt: str,
) -> BaseTool:
    """Create a Data Fabric query tool for a coded agent.

    Entity schemas are resolved lazily on the first invocation. Keep the tool
    scoped to one agent execution so its cached schema and routing cannot cross
    execution contexts.

    Pass the same outer-agent system prompt used to construct the coded agent so
    its instructions are also available to the inner SQL-generation graph.

    Args:
        llm: Language model for the inner SQL generation loop.
        name: LangChain tool name exposed to the coded agent.
        description: Description used by the outer agent to select the tool.
        entities: Data Fabric entity references available to the tool.
        base_system_prompt: Outer coded-agent system prompt forwarded to the
            inner SQL-generation graph.
    """
    entity_set = _normalize_entities(entities)
    return _build_datafabric_tool(
        _DataFabricToolConfig(
            name=name,
            description=description,
            entities=entity_set,
            resource_description=description,
            base_system_prompt=base_system_prompt,
        ),
        llm,
    )
