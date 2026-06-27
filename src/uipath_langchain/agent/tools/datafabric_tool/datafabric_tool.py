"""Data Fabric tool creation and resource detection.

This module provides an agentic ``query_datafabric`` tool with an inner
LLM sub-graph, and a ``write_datafabric`` tool for entity CRUD operations.

The read tool accepts natural language queries, runs an inner LangGraph
sub-graph for SQL generation + execution + self-correction, and
returns a natural language answer.

The write tool accepts structured write intents (insert/update/delete)
with schema-level validation for context-derived entities.

Prompt building is in ``datafabric_prompt_builder.py``.
Sub-graph definition is in ``datafabric_subgraph.py``.
"""

import asyncio
import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.platform.entities import DataFabricEntityItem

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from .compiled_ontology import CompiledOntology
from .models import (
    DataFabricQueryInput,
    DataFabricWriteInput,
    EntityWriteSchema,
)
from .write_schema_builder import build_write_tool_description
from .write_validation import (
    derive_writable_fields,
    is_entity_writable,
    validate_mutation_intent,
)

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


class DataFabricWriteHandler:
    """Manages lazy initialization and invocation of Data Fabric write operations.

    On first call, resolves entity schemas via the platform layer and builds
    EntityWriteSchema objects for context-derived entities. Subsequent calls
    reuse the cached schemas and executor.
    """

    def __init__(
        self,
        entity_set: list[DataFabricEntityItem],
    ) -> None:
        self._entity_set = entity_set
        self._write_schemas: dict[str, EntityWriteSchema] | None = None
        self._entity_id_by_key: dict[str, str] = {}
        self._write_tool_description: str | None = None
        self._compiled_ontology: CompiledOntology | None = None
        self._executor: Any | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Lazy-init: resolve entities and build write schemas on first call."""
        if self._executor is not None:
            return

        async with self._init_lock:
            if self._executor is not None:
                return

            from uipath.platform import UiPath

            from .write_executor import WriteExecutor

            sdk = UiPath()
            resolution = await sdk.entities.resolve_entity_set_async(self._entity_set)
            if not resolution.entities:
                raise ValueError(
                    "No Data Fabric entity schemas could be fetched. "
                    "Check entity identifiers and permissions."
                )

            self._write_schemas = {}
            # The LLM addresses entities by name (matching the read schema and
            # the write tool description), but the EntitiesService CRUD endpoints
            # require the entity's GUID id. Keep a name -> id map to translate at
            # execution time.
            self._entity_id_by_key = {}
            for entity in resolution.entities:
                if not is_entity_writable(entity):
                    continue
                writable = derive_writable_fields(entity)
                self._write_schemas[entity.name] = EntityWriteSchema(
                    entity_key=entity.name,
                    display_name=entity.display_name or entity.name,
                    writable_fields=writable,
                )
                if entity.id:
                    self._entity_id_by_key[entity.name] = entity.id

            # Optional ontology layer: fetch + compile the OWL ontology if the
            # platform exposes get_ontology_file_async. This method may only
            # exist on a feature branch — if it is absent we degrade gracefully
            # to the metadata-only write path (compiled_ontology stays None).
            self._compiled_ontology = await self._maybe_compile_ontology(
                resolution.entities_service
            )

            entity_access = (
                self._compiled_ontology.entity_access
                if self._compiled_ontology
                else None
            )
            self._write_tool_description = build_write_tool_description(
                self._write_schemas,
                entity_access=entity_access,
            )

            self._executor = WriteExecutor(resolution.entities_service)

    async def _maybe_compile_ontology(
        self, entities_service: Any
    ) -> CompiledOntology | None:
        """Best-effort fetch + compile of the optional OWL ontology.

        Returns the compiled ontology, or ``None`` when no ontology is
        available or the platform package does not expose the fetch method.
        Never raises — any failure degrades to the metadata-only path.
        """
        get_ontology = getattr(entities_service, "get_ontology_file_async", None)
        if not callable(get_ontology):
            logger.debug(
                "EntitiesService has no get_ontology_file_async; "
                "skipping ontology compilation (metadata-only writes)."
            )
            return None

        from .ontology_compiler import compile_ontology

        try:
            owl_turtle = await get_ontology("owl")
            if not owl_turtle:
                logger.debug("No OWL ontology returned; metadata-only writes.")
                return None
            compiled = compile_ontology(owl_turtle)
            logger.debug(
                "Compiled ontology with %d writable entities.",
                len(compiled.entity_access),
            )
            return compiled
        except Exception as exc:  # graceful no-op on any fetch/parse failure
            logger.debug("Ontology fetch/compile skipped: %s", exc)
            return None

    async def __call__(
        self,
        entity_key: str,
        operation: str,
        record_id: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Execute a write operation against a Data Fabric entity.

        Args:
            entity_key: The entity name to write to.
            operation: One of 'insert', 'update', 'delete'.
            record_id: Record ID (required for update/delete).
            fields: Field name-value pairs (required for insert/update).

        Returns:
            JSON string with the WriteResult.
        """
        logger.debug(
            "write_datafabric called: entity=%s op=%s record_id=%s",
            entity_key,
            operation,
            record_id,
        )

        await self._ensure_initialized()

        intent = DataFabricWriteInput(
            entity_key=entity_key,
            operation=operation,
            record_id=record_id,
            fields=fields,
        )

        # Validate (ontology, when present, constrains allowed operations)
        errors = validate_mutation_intent(
            intent, self._write_schemas, self._compiled_ontology
        )
        if errors:
            return json.dumps(
                {
                    "success": False,
                    "operation": operation,
                    "entity_key": entity_key,
                    "errors": errors,
                }
            )

        # Execute. The LLM addresses the entity by name, but the CRUD endpoints
        # require the entity's GUID id — translate before executing, then
        # restore the friendly name on the result for the model.
        from .write_executor import WriteExecutor

        assert isinstance(self._executor, WriteExecutor)
        resolved_key = self._entity_id_by_key.get(intent.entity_key, intent.entity_key)
        exec_intent = (
            intent.model_copy(update={"entity_key": resolved_key})
            if resolved_key != intent.entity_key
            else intent
        )
        result = await self._executor.execute(exec_intent)
        result.entity_key = intent.entity_key
        return result.model_dump_json()


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
    entity_lines = []
    for e in entity_set:
        line = f"- {e.name}"
        if e.description:
            line += f": {e.description}"
        entity_lines.append(line)
    entity_summary = "\n".join(entity_lines)

    return BaseUiPathStructuredTool(
        name=tool_name,
        description=(
            "Query the following Data Fabric entities using natural language:\n"
            f"{entity_summary}\n"
            "Describe what data you need and the tool will translate it to SQL, "
            "execute the query, and return a natural language answer."
        ),
        args_schema=DataFabricQueryInput,
        coroutine=handler,
        metadata={"tool_type": "datafabric_sql"},
    )


def _build_initial_write_tool_description(
    entity_set: list[DataFabricEntityItem],
) -> str:
    """Build a pre-resolution description for the write tool from the entity set.

    This is the description used at tool-creation time, before entity
    schemas have been lazily resolved.  It lists entity names and
    descriptions from the ``DataFabricEntityItem`` objects available
    in the agent config.  After first invocation the handler builds a
    richer field-level description via ``build_write_tool_description``.
    """
    entity_lines = []
    for e in entity_set:
        line = f"- {e.name}"
        if e.description:
            line += f": {e.description}"
        entity_lines.append(line)
    entity_summary = "\n".join(entity_lines)

    return (
        "Modify Data Fabric entities using structured operations "
        "(insert, update, delete).\n\n"
        "Available entities:\n"
        f"{entity_summary}\n\n"
        "Operations:\n"
        "- insert: provide entity_key and fields. "
        "All required fields must be included.\n"
        "- update: provide entity_key, record_id (from a prior read), "
        "and fields to change.\n"
        "- delete: provide entity_key and record_id. Requires confirmation.\n\n"
        "Query the entity first (using the read tool) to discover record IDs "
        "and current field values before updating or deleting."
    )


def create_datafabric_tools(
    resource: AgentContextResourceConfig,
    llm: BaseChatModel,
    tool_name: str = "query_datafabric",
    agent_config: dict[str, str] | None = None,
) -> list[BaseTool]:
    """Create both read and write Data Fabric tools.

    Returns a list containing:
    1. The ``query_datafabric`` read tool (NL-to-SQL subgraph)
    2. The ``write_datafabric`` write tool (structured CRUD)

    Args:
        resource: The Data Fabric context resource configuration.
        llm: The language model for the inner SQL generation loop.
        tool_name: Sanitized tool name for the read tool.
        agent_config: Optional dict with agent-level config.
    """
    # Read tool (unchanged)
    read_tool = create_datafabric_query_tool(
        resource, llm, tool_name=tool_name, agent_config=agent_config
    )

    # Write tool — always created; writability is enforced at handler level
    # after async entity resolution (entity_type / external_fields are only
    # available on resolved Entity objects, not on DataFabricEntityItem).
    entity_set = [
        DataFabricEntityItem.model_validate(item.model_dump(by_alias=True))
        for item in (resource.entity_set or [])
    ]
    write_handler = DataFabricWriteHandler(entity_set=entity_set)
    write_tool_name = f"{tool_name}_write"

    write_tool = BaseUiPathStructuredTool(
        name=write_tool_name,
        description=_build_initial_write_tool_description(entity_set),
        args_schema=DataFabricWriteInput,
        coroutine=write_handler,
        metadata={
            "tool_type": "datafabric_write",
            "require_conversational_confirmation": True,
        },
    )

    return [read_tool, write_tool]
