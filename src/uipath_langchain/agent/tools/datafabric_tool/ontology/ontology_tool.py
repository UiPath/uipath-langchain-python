"""Standalone Data Fabric Ontology tool.

The agent selects *ontologies* (not entities). On first invocation this tool:

1. fetches each ontology's **mapping** (critical — it is the entity allow-list)
   and **OWL** (optional — grounds the prompt) from the Ontology service,
2. parses the mapping into the closed entity allow-list — YARRRML (the service's
   native form: table + entity id + folder key) or R2RML Turtle
   (``(entity_name, folder_path)`` pairs),
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
from uipath.platform.entities import DataFabricEntityItem, EntitiesService, Entity

from ...base_uipath_structured_tool import BaseUiPathStructuredTool

# ``..datafabric_tool`` is the sibling *module* datafabric_tool.py (which defines
# BASE_SYSTEM_PROMPT), not the datafabric_tool package __init__.
from ..datafabric_tool import BASE_SYSTEM_PROMPT
from ..models import DataFabricQueryInput
from .ontology_fetcher import fence_ontology_block, fetch_ontology_file
from .ontology_r2rml import parse_r2rml_entities
from .ontology_yarrrml import MappedEntity, parse_yarrrml_entities

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
    mapped: list[MappedEntity] | None = None,
) -> tuple[list[Entity], EntitiesService]:
    """Resolve R2RML ``pairs`` and YARRRML ``mapped`` entities into schemas + a folder-scoped service."""
    # Step 1a: resolve each *distinct* R2RML folder path -> key, concurrently.
    distinct_paths = list(dict.fromkeys(folder_path for _, folder_path in pairs))
    resolved_keys = await asyncio.gather(
        *(sdk.folders.retrieve_key_async(folder_path=p) for p in distinct_paths)
    )
    folder_key_by_path: dict[str, str] = {}
    for folder_path, folder_key in zip(distinct_paths, resolved_keys, strict=True):
        if not folder_key:
            raise ValueError(
                f"Folder path '{folder_path}' could not be resolved to a "
                "folder key. Check the R2RML uipath:folderPath and permissions."
            )
        folder_key_by_path[folder_path] = folder_key

    wanted: list[tuple[str, str | None, str | None, str]] = [
        (name, None, folder_key_by_path[folder_path], folder_path)
        for name, folder_path in pairs
    ]
    wanted += [
        (e.name, e.entity_id, e.folder_key, e.folder_key or "<ontology folder>")
        for e in mapped or []
    ]

    # Step 1b: look up ids by name where the mapping gave none.
    async def _id_of(
        name: str, entity_id: str | None, folder_key: str | None, label: str
    ) -> str:
        if entity_id:
            return entity_id
        try:
            entity = await sdk.entities.retrieve_by_name_async(
                name, folder_key=folder_key
            )
        except Exception as e:
            raise ValueError(
                f"Failed to resolve ontology entity '{name}' in folder '{label}': {e}"
            ) from e
        return entity.id

    ids = await asyncio.gather(*(_id_of(*w) for w in wanted))

    # Step 2: delegate to the SDK's public entity-set resolver, which builds the
    # folder-scoped EntitiesService for us (no reliance on SDK-private attrs).
    items = [
        DataFabricEntityItem(id=entity_id, name=name, folder_key=folder_key)
        for (name, _, folder_key, _), entity_id in zip(wanted, ids, strict=True)
    ]
    resolution = await sdk.entities.resolve_entity_set_async(items)
    return resolution.entities, resolution.entities_service


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
            mapped: list[MappedEntity] = []

            for name, folder_key in self._ontologies:
                # The mapping is critical (it is the entity allow-list) — fetch
                # failure is fatal, not degraded.
                mapping_content, mapping_media = await fetch_ontology_file(
                    sdk.entities, name, "mapping", folder_key
                )
                r2rml_blocks.append(
                    fence_ontology_block(
                        name, "mapping", mapping_content, mapping_media
                    )
                )
                if "yaml" in mapping_media.lower():
                    mapped += [
                        e
                        for e in parse_yarrrml_entities(mapping_content, folder_key)
                        if e not in mapped
                    ]
                else:
                    pairs += [
                        p
                        for p in parse_r2rml_entities(mapping_content)
                        if p not in pairs
                    ]

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

            if not pairs and not mapped:
                raise ValueError("Ontology mapping declared no entities to query.")

            entities, scoped_service = await resolve_ontology_entities(
                sdk, pairs, mapped
            )
            if not entities:
                raise ValueError(
                    "No Data Fabric entity schemas could be resolved from the "
                    "ontology. Check the mapping's table names, folders, and "
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
