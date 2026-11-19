"""Standalone ontology context tool, grounded on and executed by the Ontology service.

The agent definition holds one ontology per ``datafabricontology`` context node,
addressed by the id the picker saved. On first invocation this tool:

1. reads the ontology from the Ontology service and refuses it unless it is
   ``DEPLOYED``,
2. grounds the inner model on the ontology — its schema files or its graph,
   selected by the ``OntologyContextSource`` flag (:mod:`ontology_grounding`),
3. builds the inner system prompt (:mod:`ontology_service_prompt`) and compiles
   the inner sub-graph (:mod:`ontology_subgraph`), whose ``execute_sql`` posts the
   model's statements to ``POST /ontology/{id}/sql`` (:mod:`ontology_service_sql`).

The model writes SQL over business objects and properties; the service applies the
mapping to the data. Nothing here reads Data Fabric directly, and the ontology id
comes only from the agent definition, never from the model.

This package shares nothing with the entity-set context tool
(``datafabric_tool``/``datafabric_subgraph``) except the ``BASE_SYSTEM_PROMPT``
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

from ...base_uipath_structured_tool import BaseUiPathStructuredTool

# ``..datafabric_tool`` is the sibling *module* datafabric_tool.py (which defines
# BASE_SYSTEM_PROMPT), not the datafabric_tool package __init__.
from ..datafabric_tool import BASE_SYSTEM_PROMPT
from ..models import DataFabricQueryInput
from .ontology_grounding import read_ontology_source, select_grounding_source
from .ontology_service_client import (
    ensure_deployed,
    ontology_label,
    resolve_ontology_client,
    validate_ontology_id,
)
from .ontology_service_prompt import build_service_prompt
from .ontology_service_sql import create_service_execute_sql_tool

if TYPE_CHECKING:
    from uipath.platform import UiPath

logger = logging.getLogger(__name__)

# Feature flag gating the whole ontology tool. Owned here (not in the entity
# tool's module) so this feature is fully self-contained; ``context_tool`` and
# the handler's defense-in-depth re-check both import it from this package.
DATAFABRIC_ONTOLOGY_FF = "DataFabricOntologyEnabled"


class DataFabricOntologyQueryHandler:
    """Lazy-init + invoke of the ontology-grounded SQL sub-graph.

    On first call, reads the ontology, checks it is deployed, grounds the prompt
    and compiles the inner sub-graph. Subsequent calls reuse the cached graph.
    """

    def __init__(
        self,
        ontology_id: str | None,
        llm: BaseChatModel,
        resource_description: str = "",
        base_system_prompt: str = "",
    ) -> None:
        self._ontology_id = ontology_id
        self._llm = llm
        self._resource_description = resource_description
        self._base_system_prompt = base_system_prompt
        self._compiled: CompiledStateGraph[Any] | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_graph(self) -> CompiledStateGraph[Any]:
        """Lazy-init: read, check, ground and compile on first call.

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

            # Defense in depth: the tool is only created when the flag is on
            # (the gate in context_tool). Re-check here — the last point before
            # any service call — so the feature can never do work with the flag
            # disabled, however the tool was constructed.
            if not FeatureFlags.is_flag_enabled(DATAFABRIC_ONTOLOGY_FF, default=False):
                raise ValueError(
                    "Data Fabric ontology tool invoked while feature flag "
                    f"'{DATAFABRIC_ONTOLOGY_FF}' is disabled."
                )

            self._compiled = await self._build_graph(UiPath())
            return self._compiled

    async def _build_graph(self, sdk: "UiPath") -> CompiledStateGraph[Any]:
        """Ground on the Ontology service and compile a sub-graph that runs SQL there."""
        from .ontology_subgraph import OntologyGraph

        ontology_id = validate_ontology_id(self._ontology_id)
        client = resolve_ontology_client(sdk)
        details = await client.get_async(ontology_id)
        ensure_deployed(details, ontology_id)
        label = ontology_label(details, ontology_id)

        source = read_ontology_source()
        grounding = await select_grounding_source(source).ground(
            client, ontology_id, details
        )

        system_prompt = build_service_prompt(
            grounding=grounding,
            details=details,
            ontology_id=ontology_id,
            resource_description=self._resource_description,
            base_system_prompt=self._base_system_prompt,
        )
        logger.info(
            "Ontology %s ('%s', %s) grounded via %s [%s]; prompt %d bytes (~%d tokens)",
            ontology_id,
            label,
            details.state,
            grounding.kind,
            grounding.summary,
            len(system_prompt.encode("utf-8")),
            len(system_prompt) // 4,
        )
        return OntologyGraph.create(
            llm=self._llm,
            system_prompt=system_prompt,
            execute_sql_tool=create_service_execute_sql_tool(
                client, ontology_id, label
            ),
        )

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
    """Create the ontology tool from a ``datafabricontology`` context.

    One node holds one ontology: ``ontology_set[0]`` is the ontology and its ``id``
    is how the tool addresses it. A missing id fails on first use with
    ``ONTOLOGY_INVALID_REFERENCE`` rather than at agent start-up, like every other
    lazy check in this tool. Extra items are reported and ignored.

    Args:
        resource: The ontology context resource configuration.
        llm: The language model for the inner SQL generation loop.
        tool_name: Tool name surfaced to the outer agent.
        agent_config: Optional dict; key ``base_system_prompt`` carries the outer
            agent's system prompt.
    """
    config = agent_config or {}
    items = list(resource.ontology_set or [])
    first = next(iter(items), None)
    if first is not None and len(items) > 1:
        logger.warning(
            "Ontology context '%s' holds %d ontologies; using the first (%s).",
            resource.name,
            len(items),
            first.name,
        )
    ontology_name = (first.name if first else None) or resource.name
    handler = DataFabricOntologyQueryHandler(
        ontology_id=getattr(first, "id", None) if first else None,
        llm=llm,
        resource_description=resource.description or "",
        base_system_prompt=config.get(BASE_SYSTEM_PROMPT, ""),
    )

    return BaseUiPathStructuredTool(
        name=tool_name,
        description=(
            f"Query the business data described by the ontology '{ontology_name}' "
            "with natural language. Describe the data you need; the tool grounds on "
            "the ontology's business objects and relationships, translates your "
            "request to SQL, runs it on the Ontology service, and returns the rows."
        ),
        args_schema=DataFabricQueryInput,
        coroutine=handler,
        metadata={"tool_type": "datafabric_ontology_sql"},
    )
