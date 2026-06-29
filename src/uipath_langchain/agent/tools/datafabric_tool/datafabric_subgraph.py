"""Inner LangGraph sub-graph for the Data Fabric agentic tool.

Implements a self-contained ReAct loop where an inner LLM translates
natural-language questions into SQL, executes them via ``execute_sql``,
and retries on errors — all within a single outer tool call.

On a successful SQL execution the graph short-circuits straight to END
rather than invoking the LLM again to reformat the records into prose;
the outer agent receives the raw tool result and produces the final
natural-language answer. Errors still loop back to the inner LLM so the
retry path remains intact.
"""

import asyncio
import logging
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from uipath.core.feature_flags import FeatureFlags
from uipath.platform.entities import EntitiesService, Entity

from ..datafabric_query_tool import DataFabricQueryTool
from . import datafabric_prompt_builder
from .models import DataFabricExecuteSqlInput
from .ontology_fetch_tool import create_ontology_fetch_tool

logger = logging.getLogger(__name__)

# Feature flag gating the Data Fabric ontology grounding feature. Defaults off:
# when disabled, the inner graph is constructed without the fetch_ontology tool
# (the original entities-only graph), so the feature stays out of the default path.
_DATAFABRIC_ONTOLOGY_FF = "DataFabricOntologyEnabled"


class DataFabricSubgraphState(BaseModel):
    """State for the inner Data Fabric ReAct sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    iteration_count: int = 0
    last_tool_success: bool = False


class QueryExecutor:
    """Executes SQL queries against Data Fabric."""

    def __init__(self, entities_service: EntitiesService) -> None:
        self._entities = entities_service

    async def __call__(self, sql_query: str) -> dict[str, Any]:
        logger.debug("execute_sql called with SQL: %s", sql_query)
        try:
            records = await self._entities.query_entity_records_async(
                sql_query=sql_query,
            )
            return {
                "records": records,
                "total_count": len(records),
                "sql_query": sql_query,
            }
        except Exception as e:
            logger.error("SQL query failed: %s", e)
            return {
                "records": [],
                "total_count": 0,
                "error": str(e),
                "sql_query": sql_query,
            }


class DataFabricGraph:
    """Inner ReAct sub-graph for Data Fabric SQL execution.

    Each graph node is a method. The graph is compiled during __init__
    and available via the ``compiled`` property.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        entities: list[Entity],
        entities_service: EntitiesService,
        max_iterations: int = 25,
        resource_description: str = "",
        base_system_prompt: str = "",
        ontologies: list[tuple[str, str | None]] | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._execute_sql_tool = self._create_execute_sql_tool(
            entities_service, entities
        )
        # Inner toolset: always execute_sql; optionally an LLM-decided
        # fetch_ontology tool, added only when ontologies are configured AND the
        # DataFabricOntologyEnabled feature flag is on. The flag decides which
        # graph gets built — off → the original entities-only graph.
        inner_tools: list[BaseTool] = [self._execute_sql_tool]
        ontology_names: list[str] = []
        if ontologies and FeatureFlags.is_flag_enabled(
            _DATAFABRIC_ONTOLOGY_FF, default=False
        ):
            inner_tools.append(
                create_ontology_fetch_tool(entities_service, ontologies)
            )
            ontology_names = [name for name, _ in ontologies]
        self._tools_by_name: dict[str, BaseTool] = {
            tool.name: tool for tool in inner_tools
        }
        # Surface the ontology in the system prompt only when its fetch tool is
        # actually bound — otherwise the LLM is told to call a tool it lacks.
        self._system_message = SystemMessage(
            content=datafabric_prompt_builder.build(
                entities,
                resource_description,
                base_system_prompt,
                ontology_names=ontology_names,
            )
        )
        self._inner_llm = llm.model_copy(update={"disable_streaming": True}).bind_tools(
            inner_tools
        )

        # Build and compile the graph
        graph = StateGraph(DataFabricSubgraphState)
        graph.add_node("inner_llm", self.llm_node)
        graph.add_node("inner_tool", self.tool_node)
        graph.add_node("termination", self.termination_node)
        graph.add_edge(START, "inner_llm")
        graph.add_conditional_edges(
            "inner_llm", self.router, ["inner_tool", "termination", END]
        )
        graph.add_conditional_edges("inner_tool", self.tool_router, ["inner_llm", END])
        graph.add_edge("termination", END)
        self.compiled_graph: CompiledStateGraph[Any] = graph.compile()

    async def llm_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Invoke the inner LLM with the current message history."""
        messages = [self._system_message] + list(state.messages)
        response = await self._inner_llm.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Execute all tool calls from the last AIMessage concurrently."""
        last = state.messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"iteration_count": state.iteration_count}

        results = await asyncio.gather(
            *[self._execute_tool_call(tc) for tc in last.tool_calls]
        )
        # End as soon as ANY tool call is a terminal success (a row-returning
        # execute_sql). `any` not `all`: a non-terminal tool (e.g. fetch_ontology)
        # co-issued in the same turn must not prevent a successful SQL from ending
        # the loop.
        any_succeeded = any(success for _, success in results)
        # When short-circuiting to END, return ONLY the terminal-success
        # ToolMessages so the outer agent's result is the query rows — not a
        # co-issued fetch_ontology's OWL. On a non-terminal turn keep all messages
        # so the inner LLM can use them on its next pass.
        if any_succeeded:
            tool_messages = [msg for msg, success in results if success]
        else:
            tool_messages = [msg for msg, _ in results]
        return {
            "messages": tool_messages,
            "iteration_count": state.iteration_count + len(last.tool_calls),
            "last_tool_success": any_succeeded,
        }

    async def _execute_tool_call(self, tool_call: ToolCall) -> tuple[ToolMessage, bool]:
        """Execute a single tool call and report whether it is a terminal success.

        Dispatches by tool name so the sub-graph can host more than one tool
        (e.g. ``execute_sql`` and ``fetch_ontology``). Only a successful
        ``execute_sql`` that returned rows is terminal; every other tool
        (including ontology fetch) reports ``False`` so the router loops back to
        the inner LLM, letting it use the result to write or refine SQL.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        tool = self._tools_by_name.get(name)
        if tool is None:
            return (
                ToolMessage(
                    content=f"Unknown tool: {name}",
                    tool_call_id=tool_call["id"],
                    name=name,
                ),
                False,
            )
        try:
            result = await tool.ainvoke(args)
        except ValueError as e:
            if name == self._execute_sql_tool.name:
                result = {
                    "records": [],
                    "total_count": 0,
                    "error": str(e),
                    "sql_query": args.get("sql_query", ""),
                }
            else:
                result = f"Tool '{name}' failed: {e}"
        succeeded = (
            name == self._execute_sql_tool.name
            and isinstance(result, dict)
            and not result.get("error")
            and result.get("total_count", 0) > 0
        )
        return (
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=name,
            ),
            succeeded,
        )

    async def termination_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Produce a clear message when max iterations is reached."""
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I was unable to resolve the query after "
                        f"{state.iteration_count} SQL attempts. "
                        "Please try rephrasing the question or narrowing the scope."
                    )
                )
            ]
        }

    def router(self, state: DataFabricSubgraphState) -> str:
        """Route from ``inner_llm`` to tool, termination, or END."""
        last = state.messages[-1] if state.messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            if state.iteration_count < self._max_iterations:
                return "inner_tool"
            return "termination"
        return END

    def tool_router(self, state: DataFabricSubgraphState) -> str:
        """Route from ``inner_tool``: short-circuit on success, retry on error.

        Skips the redundant LLM call that would otherwise reformat a
        successful SQL result into prose — the outer agent receives the
        raw tool output and produces the final natural-language answer.
        Errors loop back to ``inner_llm`` so the retry path is preserved.
        """
        if state.last_tool_success:
            return END
        return "inner_llm"

    def _create_execute_sql_tool(
        self,
        entities_service: EntitiesService,
        entities: list[Entity],
    ) -> BaseTool:
        """Create the inner ``execute_sql`` tool."""
        entity_names = ", ".join(e.name for e in entities)
        return DataFabricQueryTool(
            name="execute_sql",
            description=(
                f"Execute a SQL SELECT query against Data Fabric entities: {entity_names}. "
                "Refer to the entity schemas in the system message for available "
                "tables and columns. Retry with a corrected query on errors."
            ),
            args_schema=DataFabricExecuteSqlInput,
            coroutine=QueryExecutor(entities_service),
            metadata={"tool_type": "datafabric_sql"},
        )

    @staticmethod
    def create(
        llm: BaseChatModel,
        entities: list[Entity],
        entities_service: EntitiesService,
        max_iterations: int = 25,
        resource_description: str = "",
        base_system_prompt: str = "",
        ontologies: list[tuple[str, str | None]] | None = None,
    ) -> CompiledStateGraph[Any]:
        """Create and return a compiled Data Fabric sub-graph."""
        graph = DataFabricGraph(
            llm,
            entities,
            entities_service,
            max_iterations,
            resource_description,
            base_system_prompt,
            ontologies,
        )
        return graph.compiled_graph
