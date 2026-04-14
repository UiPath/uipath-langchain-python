"""Inner LangGraph sub-graph for the Data Fabric agentic tool.

Implements a self-contained ReAct loop where an inner LLM translates
natural-language questions into SQL, executes them via ``execute_sql``,
and retries on errors — all within a single outer tool call.
"""

import asyncio
import json
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
from uipath.platform.entities import EntitiesService, Entity

from ..datafabric_query_tool import DataFabricQueryTool
from . import datafabric_prompt_builder
from .models import DataFabricExecuteSqlInput

logger = logging.getLogger(__name__)


class DataFabricSubgraphState(BaseModel):
    """State for the inner Data Fabric ReAct sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    iteration_count: int = 0
    last_sql: str | None = None
    repeated_sql_count: int = 0


_ERROR_HINTS: list[tuple[str, str]] = [
    ("limit", "Add a LIMIT clause to the query."),
    ("select *", "Specify explicit column names instead of SELECT *."),
    ("count(*)", "Use COUNT(column_name) instead of COUNT(*)."),
    ("count(1)", "Use COUNT(column_name) instead of COUNT(1)."),
    ("unknown column", "Check column names against the entity schema above."),
    ("no such column", "Check column names against the entity schema above."),
    ("no such table", "Check table names against the entity schema above."),
    ("right join", "Only LEFT JOIN is supported."),
    ("full outer join", "Only LEFT JOIN is supported."),
    ("cross join", "Only LEFT JOIN is supported."),
    ("syntax error", "Check SQL syntax — ensure proper quoting and clause order."),
]


def _classify_error_hint(error_msg: str) -> str:
    """Map a SQL error message to an actionable hint for the LLM."""
    lower = error_msg.lower()
    for pattern, hint in _ERROR_HINTS:
        if pattern in lower:
            return hint
    return "Review the error and adjust the query accordingly."


def _format_tool_result(result: dict[str, Any]) -> str:
    """Format a SQL execution result as structured JSON for the LLM.

    Provides clear status, row counts, sample data, and actionable hints
    on errors — enabling targeted self-correction (SQL-CRAFT inspired).
    """
    structured: dict[str, Any] = {
        "status": "error" if result.get("error") else "success",
        "sql_query": result.get("sql_query", ""),
    }
    if result.get("error"):
        structured["error_message"] = result["error"]
        structured["hint"] = _classify_error_hint(result["error"])
    else:
        records = result.get("records", [])
        structured["row_count"] = result.get("total_count", len(records))
        structured["sample_rows"] = records[:5]
        # Sanity-check hints
        if structured["row_count"] == 0:
            structured["hint"] = (
                "Query returned 0 results. Verify filter conditions "
                "are not too restrictive or column values are correct."
            )
    return json.dumps(structured, default=str)


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
    ) -> None:
        self._max_iterations = max_iterations
        self._execute_sql_tool = self._create_execute_sql_tool(
            entities_service, entities
        )
        self._system_message = SystemMessage(
            content=datafabric_prompt_builder.build(
                entities, resource_description, base_system_prompt
            )
        )
        self._inner_llm = llm.model_copy(update={"disable_streaming": True}).bind_tools(
            [self._execute_sql_tool]
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
        graph.add_edge("inner_tool", "inner_llm")
        graph.add_edge("termination", END)
        self.compiled_graph: CompiledStateGraph[Any] = graph.compile()

    async def llm_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Invoke the inner LLM with the current message history."""
        messages = [self._system_message] + list(state.messages)
        response = await self._inner_llm.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Execute all tool calls from the last AIMessage concurrently.

        Tracks repeated SQL queries and terminates early if the LLM keeps
        retrying the same failing query (convergence detection).
        """
        last = state.messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"iteration_count": state.iteration_count}

        # Convergence detection: check if SQL is identical to last attempt
        current_sql = last.tool_calls[0].get("args", {}).get("sql_query", "")
        repeated = state.repeated_sql_count
        last_sql = state.last_sql
        if current_sql and current_sql == last_sql:
            repeated += 1
        else:
            repeated = 0

        if repeated >= 2:
            logger.warning("Convergence detected: same SQL repeated %d times", repeated)
            return {
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "error",
                                "error_message": (
                                    "The same query has been attempted multiple times "
                                    "with the same error. Try a fundamentally different "
                                    "approach or simplify the query."
                                ),
                                "hint": "Rewrite the query using a different strategy.",
                            }
                        ),
                        tool_call_id=last.tool_calls[0]["id"],
                        name="execute_sql",
                    )
                ],
                "iteration_count": state.iteration_count + 1,
                "last_sql": current_sql,
                "repeated_sql_count": repeated,
            }

        tool_messages = await asyncio.gather(
            *[self._execute_tool_call(tc) for tc in last.tool_calls]
        )
        return {
            "messages": list(tool_messages),
            "iteration_count": state.iteration_count + len(last.tool_calls),
            "last_sql": current_sql,
            "repeated_sql_count": repeated,
        }

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        """Execute a single tool call and wrap the result as structured JSON."""
        args = tool_call.get("args", {})
        try:
            result = await self._execute_sql_tool.ainvoke(args)
        except ValueError as e:
            result = {
                "records": [],
                "total_count": 0,
                "error": str(e),
                "sql_query": args.get("sql_query", ""),
            }
        return ToolMessage(
            content=_format_tool_result(result),
            tool_call_id=tool_call["id"],
            name="execute_sql",
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
        """Route to tool, termination, or END based on state."""
        last = state.messages[-1] if state.messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            if state.iteration_count < self._max_iterations:
                return "inner_tool"
            return "termination"
        return END

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
    ) -> CompiledStateGraph[Any]:
        """Create and return a compiled Data Fabric sub-graph."""
        graph = DataFabricGraph(
            llm,
            entities,
            entities_service,
            max_iterations,
            resource_description,
            base_system_prompt,
        )
        return graph.compiled_graph
