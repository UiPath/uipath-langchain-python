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
from contextlib import contextmanager
from typing import Annotated, Any, Iterator

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
from uipath.platform.errors import EnrichedException

try:
    from uipath.platform.errors import DataFabricError
except ImportError:
    DataFabricError = None  # type: ignore[assignment,misc]

from ..datafabric_query_tool import DataFabricQueryTool
from . import datafabric_prompt_builder
from .models import DataFabricExecuteSqlInput

logger = logging.getLogger(__name__)
CATEGORY_MARKER = "(category: "


@contextmanager
def _noop_context() -> Iterator[None]:
    yield None


class DataFabricSubgraphState(BaseModel):
    """State for the inner Data Fabric ReAct sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    iteration_count: int = 0
    last_tool_success: bool = False
    last_error_category: str = ""
    last_error_detail: str = ""


class QueryExecutor:
    """Executes SQL queries against Data Fabric."""

    def __init__(
        self, entities_service: EntitiesService, entities: list[Entity]
    ) -> None:
        self._entities = entities_service
        native = [e.name for e in entities if not e.external_fields]
        federated = [e.name for e in entities if e.external_fields]
        self._entity_attrs: dict[str, str | int] = {
            "df.entity_count": len(entities),
            "df.native_entity_count": len(native),
            "df.federated_entity_count": len(federated),
            "df.native_entities": ", ".join(native) if native else "",
            "df.federated_entities": ", ".join(federated) if federated else "",
        }

    async def __call__(self, sql_query: str) -> dict[str, Any]:
        logger.debug("execute_sql called with SQL: %s", sql_query)

        try:
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("uipath_langchain.datafabric")
        except ImportError:
            tracer = None

        span_ctx = (
            tracer.start_as_current_span(
                "Data Fabric SQL query",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "span_type": "datafabricQuery",
                    "uipath.custom_instrumentation": True,
                    "df.sql_query": sql_query,
                    **self._entity_attrs,
                },
            )
            if tracer
            else _noop_context()
        )

        with span_ctx as span:
            try:
                records = await self._entities.query_entity_records_async(
                    sql_query=sql_query,
                )
                if span is not None:
                    span.set_attribute("df.record_count", len(records))
                    span.set_attribute("df.success", True)
                return {
                    "records": records,
                    "total_count": len(records),
                    "sql_query": sql_query,
                }
            except Exception as e:
                return self._handle_query_error(e, span, sql_query)

    def _handle_query_error(
        self, e: Exception, span: Any, sql_query: str
    ) -> dict[str, Any]:
        """Handle a failed SQL query: log, record span attributes, return error dict."""
        logger.error("SQL query failed: %s", e)

        df_error = None
        if isinstance(e, EnrichedException) and DataFabricError is not None:
            df_error = DataFabricError.from_enriched_exception(e)

        if span is not None:
            self._record_error_span(span, e, df_error)

        return {
            "records": [],
            "total_count": 0,
            "error": self._build_error_detail(e, df_error),
            "sql_query": sql_query,
        }

    @staticmethod
    def _record_error_span(
        span: Any, e: Exception, df_error: "DataFabricError | None"
    ) -> None:
        """Set error attributes on an OTEL span."""
        span.set_attribute("df.success", False)
        span.set_attribute("df.error.raw", str(e)[:500])
        if df_error:
            if df_error.code:
                span.set_attribute("df.error.code", df_error.code)
            if df_error.message:
                span.set_attribute("df.error.message", df_error.message)
            if df_error.trace_id:
                span.set_attribute("df.error.trace_id", df_error.trace_id)
            span.set_attribute("df.error.category", df_error.category.value)

        from opentelemetry.trace import StatusCode

        span.record_exception(e)
        span.set_status(StatusCode.ERROR, str(e)[:200])

    @staticmethod
    def _build_error_detail(exc: Exception, df_error: "DataFabricError | None") -> str:
        """Build a structured error string for the inner LLM."""
        if df_error and df_error.code:
            parts = [f"[{df_error.code}]"]
            if df_error.category.value != "unknown":
                parts.append(f"(category: {df_error.category.value})")
            if df_error.message:
                parts.append(df_error.message)
            if df_error.is_retryable:
                parts.append("— This error is transient, retry the same query.")
            elif df_error.is_bad_sql:
                parts.append("— Fix the SQL syntax and retry.")
            return " ".join(parts)
        return str(exc)


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
        tool_messages = [msg for msg, _, _, _ in results]
        all_succeeded = bool(results) and all(ok for _, ok, _, _ in results)

        # Capture last error info from the most recent failed call
        last_category = ""
        last_detail = ""
        for _, ok, cat, detail in reversed(results):
            if not ok and detail:
                last_category = cat
                last_detail = detail
                break

        return {
            "messages": tool_messages,
            "iteration_count": state.iteration_count + len(last.tool_calls),
            "last_tool_success": all_succeeded,
            "last_error_category": last_category or state.last_error_category,
            "last_error_detail": last_detail or state.last_error_detail,
        }

    async def _execute_tool_call(
        self, tool_call: ToolCall
    ) -> tuple[ToolMessage, bool, str, str]:
        """Execute a single tool call and report whether it succeeded.

        Returns (message, succeeded, error_category, error_detail).
        """
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
        error_str = result.get("error", "") if isinstance(result, dict) else ""
        succeeded = (
            isinstance(result, dict)
            and not error_str
            and result.get("total_count", 0) > 0
        )
        # Extract category from structured error like "[SQL_VALIDATION] (category: bad_sql) ..."
        error_category = ""
        if error_str and CATEGORY_MARKER in error_str:
            start = error_str.index(CATEGORY_MARKER) + len(CATEGORY_MARKER)
            end = error_str.index(")", start)
            error_category = error_str[start:end]
        return (
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name="execute_sql",
            ),
            succeeded,
            error_category,
            error_str,
        )

    async def termination_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Produce a clear message when max iterations is reached."""
        parts = [
            f"I was unable to resolve the query after "
            f"{state.iteration_count} SQL attempts.",
        ]
        if state.last_error_category:
            parts.append(f"Last error category: {state.last_error_category}.")
        if state.last_error_detail:
            parts.append(f"Last error: {state.last_error_detail[:300]}")
        parts.append("Please try rephrasing the question or narrowing the scope.")
        return {"messages": [AIMessage(content=" ".join(parts))]}

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
            coroutine=QueryExecutor(entities_service, entities),
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
