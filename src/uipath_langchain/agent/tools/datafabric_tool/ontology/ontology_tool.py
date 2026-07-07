"""Standalone Data Fabric Ontology tool.

The agent selects *ontologies* (not entities). On first invocation this tool:

1. fetches each ontology's **R2RML** (critical — it is the entity allow-list) and
   **OWL** (optional — grounds the prompt),
2. parses R2RML into the closed ``(entity_name, folder_path)`` allow-list,
3. resolves each entity to its schema and builds a folder-scoped
   ``EntitiesService`` (see :func:`resolve_ontology_entities`),
4. compiles the ontology sub-graph (:class:`ontology_subgraph.DataFabricGraph`),
   grounded on both the OWL and the R2RML.

Everything from the sub-graph down (``execute_sql`` → ``query_entity_records_async``)
mirrors the entity tool. Ontology names/folders come only from the agent
definition; the LLM never chooses which entity or folder to reach.

This package duplicates the sub-graph and prompt builder so the entity tool's
``datafabric_tool``/``datafabric_subgraph`` are left untouched by this feature.
The only symbol borrowed from the entity tool is the shared ``BASE_SYSTEM_PROMPT``
agent-config key, so both tools read the outer prompt from the same place.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.platform.entities import EntitiesService, Entity

from ...base_uipath_structured_tool import BaseUiPathStructuredTool
from ..datafabric_tool import BASE_SYSTEM_PROMPT
from ..models import DataFabricQueryInput
from .ontology_fetcher import fence_ontology_block, fetch_ontology_file
from .ontology_r2rml import parse_r2rml_entities

if TYPE_CHECKING:
    from uipath.platform import UiPath

logger = logging.getLogger(__name__)

# Feature flag gating the whole ontology tool. Owned here (not in the entity
# tool's module) so this feature is fully self-contained; ``context_tool`` and
# the handler's defense-in-depth re-check both import it from this package.
DATAFABRIC_ONTOLOGY_FF = "DataFabricOntologyEnabled"


async def resolve_ontology_entities(
    sdk: "UiPath",
    pairs: list[tuple[str, str]],
) -> tuple[list[Entity], EntitiesService]:
    """Resolve an R2RML allow-list into schemas + a folder-scoped service (b2).

    For each ``(entity_name, folder_path)``: resolve the folder path to a key
    (cached per distinct path), fetch the entity schema by name within that
    folder, and accumulate a ``folders_map`` used to build an ``EntitiesService``
    that routes each entity's record queries to its own folder.

    Folder keys come only from the trusted folder resolution of the R2RML
    ``uipath:folderPath`` — never from the LLM.
    """
    entities: list[Entity] = []
    folders_map: dict[str, str] = {}
    folder_key_cache: dict[str, str] = {}

    for name, folder_path in pairs:
        folder_key = folder_key_cache.get(folder_path)
        if folder_key is None:
            folder_key = await sdk.folders.retrieve_key_async(folder_path=folder_path)
            if not folder_key:
                raise ValueError(
                    f"Folder path '{folder_path}' could not be resolved to a "
                    "folder key. Check the R2RML uipath:folderPath and permissions."
                )
            folder_key_cache[folder_path] = folder_key

        entity = await sdk.entities.retrieve_by_name_async(name, folder_key=folder_key)
        entities.append(entity)
        folders_map[entity.name] = folder_key

    # Build the folder-scoped service ourselves via the public folders_map param
    # (yields a FoldersMapRoutingStrategy). Reuse the SDK's already-resolved
    # auth/config so this service talks to the same tenant as sdk.entities.
    scoped_service = EntitiesService(
        config=sdk._config,
        execution_context=sdk._execution_context,
        folders_service=sdk.folders,
        folders_map=folders_map,
    )
    return entities, scoped_service


class DataFabricOntologyQueryHandler:
    """Lazy-init + invoke of the ontology-grounded Data Fabric sub-graph.

    On first call, fetches OWL + R2RML, derives the entity allow-list from R2RML,
    resolves schemas + a folder-scoped service, and compiles the inner sub-graph.
    Subsequent calls reuse the cached graph.
    """

    def __init__(
        self,
        ontologies: list[tuple[str, str | None]],
        llm: BaseChatModel,
        resource_description: str = "",
        base_system_prompt: str = "",
    ) -> None:
        self._ontologies = ontologies
        self._llm = llm
        self._resource_description = resource_description
        self._base_system_prompt = base_system_prompt
        self._compiled: CompiledStateGraph[Any] | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_graph(self) -> CompiledStateGraph[Any]:
        """Lazy-init: fetch + parse + resolve + compile on first call.

        Uses ``asyncio.Lock`` because the outer agent supports parallel tool
        calls — two concurrent invocations could otherwise race on first call.
        """
        if self._compiled is not None:
            return self._compiled

        async with self._init_lock:
            if self._compiled is not None:
                return self._compiled

            from uipath.core.feature_flags import FeatureFlags
            from uipath.platform import UiPath

            from . import ontology_prompt_builder
            from .ontology_subgraph import DataFabricGraph

            # Defense in depth: the tool is only created when the flag is on
            # (the gate in context_tool). Re-check here — the last point before
            # any ontology fetch/parse/resolution — so the feature can never do
            # work with the flag disabled, however the tool was constructed.
            if not FeatureFlags.is_flag_enabled(DATAFABRIC_ONTOLOGY_FF, default=False):
                raise ValueError(
                    "Data Fabric ontology tool invoked while feature flag "
                    f"'{DATAFABRIC_ONTOLOGY_FF}' is disabled."
                )

            sdk = UiPath()

            owl_blocks: list[str] = []
            r2rml_blocks: list[str] = []
            pairs: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()

            for name, folder_key in self._ontologies:
                # R2RML is critical (it is the entity allow-list) — fetch failure
                # is fatal, not degraded.
                r2rml_content, r2rml_media = await fetch_ontology_file(
                    sdk.entities, name, "r2rml", folder_key
                )
                r2rml_blocks.append(
                    fence_ontology_block(name, "r2rml", r2rml_content, r2rml_media)
                )
                for pair in parse_r2rml_entities(r2rml_content):
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)

                # OWL only grounds the prompt — degrade to a note on failure.
                try:
                    owl_content, owl_media = await fetch_ontology_file(
                        sdk.entities, name, "owl", folder_key
                    )
                    owl_blocks.append(
                        fence_ontology_block(name, "owl", owl_content, owl_media)
                    )
                except Exception as e:
                    logger.warning("Ontology OWL fetch failed for %r: %s", name, e)
                    owl_blocks.append(
                        f"Ontology '{name}' OWL is unavailable ({type(e).__name__}); "
                        "rely on the R2RML mapping and entity schemas below."
                    )

            if not pairs:
                raise ValueError("Ontology R2RML declared no entities to query.")

            entities, scoped_service = await resolve_ontology_entities(sdk, pairs)
            if not entities:
                raise ValueError(
                    "No Data Fabric entity schemas could be resolved from the "
                    "ontology. Check the R2RML table names, folder paths, and "
                    "permissions."
                )

            system_prompt = ontology_prompt_builder.build(
                entities,
                resource_description=self._resource_description,
                base_system_prompt=self._base_system_prompt,
                ontology_text="\n\n".join(owl_blocks),
                r2rml_text="\n\n".join(r2rml_blocks),
            )
            self._compiled = DataFabricGraph.create(
                llm=self._llm,
                entities=entities,
                entities_service=scoped_service,
                system_prompt=system_prompt,
            )
            return self._compiled

    async def __call__(self, user_query: str) -> str:
        logger.debug("query_datafabric_ontology called with: %s", user_query)

        compiled_graph = await self._ensure_graph()
        result_state = await compiled_graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}
        )
        messages = result_state["messages"]
        last_message = messages[-1] if messages else None

        # Happy path: the sub-graph short-circuits at END after a successful
        # execute_sql, leaving a trailing batch of ToolMessages. Collapse the
        # trailing batch into one synthetic message so the outer agent sees the
        # full result set.
        if isinstance(last_message, ToolMessage):
            trailing_tool_messages: list[ToolMessage] = []
            for msg in reversed(messages):
                if not isinstance(msg, ToolMessage):
                    break
                trailing_tool_messages.append(msg)
            return self._format_terminal_tool_messages(
                list(reversed(trailing_tool_messages))
            )

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


def create_datafabric_ontology_tool(
    resource: AgentContextResourceConfig,
    llm: BaseChatModel,
    tool_name: str = "query_datafabric_ontology",
    agent_config: dict[str, str] | None = None,
) -> BaseTool:
    """Create the standalone ontology tool from a ``datafabricontology`` context.

    Args:
        resource: The Data Fabric ontology context resource configuration; its
            ``ontology_set`` supplies the ``(name, folder_key)`` pairs.
        llm: The language model for the inner SQL generation loop.
        tool_name: Tool name surfaced to the outer agent.
        agent_config: Optional dict; key ``base_system_prompt`` carries the outer
            agent's system prompt.
    """
    config = agent_config or {}
    ontologies: list[tuple[str, str | None]] = [
        (item.name, item.folder_key) for item in (resource.ontology_set or [])
    ]
    handler = DataFabricOntologyQueryHandler(
        ontologies=ontologies,
        llm=llm,
        resource_description=resource.description or "",
        base_system_prompt=config.get(BASE_SYSTEM_PROMPT, ""),
    )
    ontology_summary = "\n".join(f"- {name}" for name, _ in ontologies)

    return BaseUiPathStructuredTool(
        name=tool_name,
        description=(
            "Query UiPath Data Fabric using the following ontologies with natural "
            "language:\n"
            f"{ontology_summary}\n"
            "Describe the data you need; the tool grounds on each ontology and its "
            "mapping, translates your request to SQL, executes it, and returns a "
            "natural language answer."
        ),
        args_schema=DataFabricQueryInput,
        coroutine=handler,
        metadata={"tool_type": "datafabric_ontology_sql"},
    )
