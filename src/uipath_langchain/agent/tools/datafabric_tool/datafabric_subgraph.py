"""Inner LangGraph sub-graph for the Data Fabric agentic tool.

Implements a self-contained ReAct loop where an inner LLM translates
natural-language questions into SQL, executes them via ``execute_sql``,
and retries on errors — all within a single outer tool call. When the
entities declare operations, ``execute_operation`` runs them too.

On a successful SQL execution the graph short-circuits straight to END
rather than invoking the LLM again to reformat the records into prose;
the outer agent receives the raw tool result and produces the final
natural-language answer. Errors still loop back to the inner LLM so the
retry path remains intact.
"""

import asyncio
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Iterator

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
from pydantic import BaseModel, ValidationError
from uipath.platform.entities import EntitiesService, Entity
from uipath.platform.errors import DataFabricError, DataFabricErrorCategory

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from ..datafabric_query_tool import DataFabricQueryTool
from . import datafabric_prompt_builder
from .models import (
    EXECUTE_OPERATION,
    EXECUTE_SQL,
    MUTATION_KIND,
    READ_KIND,
    DataFabricExecuteOperationInput,
    DataFabricExecuteSqlInput,
    OperationArgument,
    entity_operations,
    operation_arguments,
    operation_kind,
)

if TYPE_CHECKING:
    from uipath.platform.entities import EntityOperation

logger = logging.getLogger(__name__)
CATEGORY_MARKER = "(category: "
# Outcomes that answer the request. Refused and Faulted go back to the inner LLM,
# unless the call was a Mutation that may have written.
OPERATION_SUCCESS_OUTCOMES = frozenset({"Returned", "Wrote", "NoChange"})
# The only outcomes of a sent Mutation that are known to have written nothing.
MUTATION_UNWRITTEN_OUTCOMES = frozenset({"Refused", "NoChange"})
MAX_MUTATIONS_PER_RUN = 2


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
    allow_changes: bool = False
    mutation_count: int = 0
    sent_mutations: list[str] = []


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
                    relationships_as_scalar=True,
                    resolve_choice_sets=True,
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

        # Covers both origins: client-side validation rejections raised before
        # the request, and errors returned by the query engine.
        df_error = DataFabricError.from_exception(e)

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

        from opentelemetry.trace import Status, StatusCode

        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)[:200]))

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
            elif df_error.category == DataFabricErrorCategory.UNSUPPORTED_CONSTRUCT:
                parts.append(
                    "— The entity query engine cannot express this construct. Do NOT "
                    "retry a variant of the same shape (another subquery, UNION, or "
                    "CTE). Rewrite it as a single flat SELECT with explicit JOINs, or "
                    "if the question cannot be expressed that way, stop calling "
                    "execute_sql and say so in a plain text reply."
                )
            return " ".join(parts)
        return str(exc)


@dataclass
class MutationBudget:
    """Mutation invokes left in one outer call, shared by a batch of tool calls."""

    remaining: int
    # The canonical form of each Mutation sent in this outer call.
    sent: list[str] = field(default_factory=list)
    # Set once a sent Mutation may have written, which ends the inner loop.
    may_have_written: bool = False

    def take(self) -> bool:
        """Claim one invoke; False when none are left."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


class OperationExecutor:
    """Runs the operations Data Fabric entities declare."""

    def __init__(
        self, entities_service: EntitiesService, entities: list[Entity]
    ) -> None:
        self._entities = entities_service
        self._by_name = {e.name: e for e in entities if entity_operations(e)}

    async def __call__(
        self,
        entity_name: str,
        operation_name: str,
        arguments: list[OperationArgument] | None = None,
        *,
        allow_changes: bool = False,
        budget: MutationBudget | None = None,
    ) -> dict[str, Any]:
        """Check the call against the declared operations, then invoke it once."""
        entity = self._by_name.get(entity_name)
        if entity is None:
            return {
                "error": f"Entity '{entity_name}' declares no operations.",
                "valid_entities": list(self._by_name),
            }

        operations = {op.name: op for op in entity_operations(entity)}
        operation = operations.get(operation_name)
        if operation is None:
            return {
                "entity": entity_name,
                "error": f"'{entity_name}' declares no operation '{operation_name}'.",
                "valid_operations": list(operations),
            }

        call = {"entity": entity_name, "operation": operation_name}
        kind = operation_kind(operation)
        values = operation_arguments(arguments or [], operation)
        missing = [
            p.name
            for p in operation.parameters
            if p.is_required and p.name not in values
        ]
        if missing:
            return {
                **call,
                "error": f"Missing required parameters: {', '.join(missing)}.",
                "parameters": [
                    {
                        "name": p.name,
                        "sql_type": p.sql_type,
                        "is_required": p.is_required,
                        "is_list": p.is_list,
                    }
                    for p in operation.parameters
                ],
            }

        if kind == MUTATION_KIND:
            if not allow_changes:
                return {
                    **call,
                    "error": (
                        f"'{operation_name}' changes data, and this request does "
                        "not allow changes."
                    ),
                    "read_operations": [
                        op.name
                        for op in operations.values()
                        if operation_kind(op) == READ_KIND
                    ],
                }
            if budget is not None:
                sent = json.dumps(
                    [entity.name, operation.name, values], sort_keys=True, default=str
                )
                if sent in budget.sent:
                    return {
                        **call,
                        "error": (
                            "This change was already sent in this request, so it "
                            "is not sent again."
                        ),
                    }
                if not budget.take():
                    return {
                        **call,
                        "error": (
                            f"At most {MAX_MUTATIONS_PER_RUN} operations that change "
                            "data run per request."
                        ),
                    }
                # Recorded before the await, so a repeat in the same batch is refused.
                budget.sent.append(sent)

        result = await self._invoke(entity, operation, kind, values)
        if (
            kind == MUTATION_KIND
            and budget is not None
            and result.get("outcome") not in MUTATION_UNWRITTEN_OUTCOMES
        ):
            budget.may_have_written = True
        return result

    async def _invoke(
        self,
        entity: Entity,
        operation: "EntityOperation",
        kind: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the operation inside an OTEL span; errors are returned as data."""
        try:
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("uipath_langchain.datafabric")
        except ImportError:
            tracer = None

        span_ctx = (
            tracer.start_as_current_span(
                "Data Fabric operation",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "span_type": "datafabricOperation",
                    "uipath.custom_instrumentation": True,
                    "df.entity": entity.name,
                    "df.operation": operation.name,
                    "df.operation_kind": kind,
                },
            )
            if tracer
            else _noop_context()
        )

        call = {"entity": entity.name, "operation": operation.name, "kind": kind}
        with span_ctx as span:
            try:
                # The resolution service routes the entity to its folder.
                result = await self._entities.invoke_operation_async(
                    entity.name, operation.name, arguments
                )
            except Exception as e:
                return self._handle_invoke_error(e, span, call)
            if span is not None:
                span.set_attribute("df.outcome", result.outcome)
                span.set_attribute("df.rows_affected", result.rows_affected)
                span.set_attribute(
                    "df.success", result.outcome in OPERATION_SUCCESS_OUTCOMES
                )
                if result.invocation_id is not None:
                    span.set_attribute("df.invocation_id", result.invocation_id)
        return {
            **call,
            "outcome": result.outcome,
            "rows_affected": result.rows_affected,
            "rows": result.rows,
            "result": result.result,
            "errors": result.errors,
            "invocation_id": result.invocation_id,
            "prints": result.prints,
            "withheld": result.withheld,
        }

    @staticmethod
    def _handle_invoke_error(
        e: Exception, span: Any, call: dict[str, Any]
    ) -> dict[str, Any]:
        """Log a failed invoke, record it on the span and return it as data."""
        logger.error("Entity operation failed: %s", e)
        df_error = DataFabricError.from_exception(e)
        if span is not None:
            QueryExecutor._record_error_span(span, e, df_error)

        if df_error and df_error.code:
            detail = " ".join(filter(None, [f"[{df_error.code}]", df_error.message]))
        else:
            detail = str(e)
        if call["kind"] == MUTATION_KIND:
            # The invoke is never retried, but a failed response can still follow
            # an applied write.
            detail += " The change may already have been applied; do not run it again."
        return {**call, "error": detail}


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
        tools = [self._execute_sql_tool]
        self._operation_executor: OperationExecutor | None = None
        if any(entity_operations(e) for e in entities):
            self._operation_executor = OperationExecutor(entities_service, entities)
            tools.append(
                self._create_execute_operation_tool(self._operation_executor, entities)
            )
        self._tool_names = [tool.name for tool in tools]
        self._system_message = SystemMessage(
            content=datafabric_prompt_builder.build(
                entities,
                resource_description,
                base_system_prompt,
                entities_service=entities_service,
            )
        )
        self._inner_llm = llm.model_copy(update={"disable_streaming": True}).bind_tools(
            tools
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

        budget = MutationBudget(
            MAX_MUTATIONS_PER_RUN - state.mutation_count,
            sent=list(state.sent_mutations),
        )
        results = await asyncio.gather(
            *[
                self._execute_tool_call(tc, state.allow_changes, budget)
                for tc in last.tool_calls
            ]
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
            # A Mutation that may have written ends the loop even when a sibling
            # call failed, so the inner LLM cannot send it again.
            "last_tool_success": all_succeeded or budget.may_have_written,
            "last_error_category": last_category or state.last_error_category,
            "last_error_detail": last_detail or state.last_error_detail,
            "mutation_count": MAX_MUTATIONS_PER_RUN - budget.remaining,
            "sent_mutations": budget.sent,
        }

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        allow_changes: bool,
        budget: MutationBudget,
    ) -> tuple[ToolMessage, bool, str, str]:
        """Dispatch a single tool call by name and report whether it succeeded.

        Returns (message, succeeded, error_category, error_detail).
        """
        name = tool_call["name"]
        # Without operations every call runs as SQL, whatever its name.
        if name == EXECUTE_SQL or self._operation_executor is None:
            return await self._execute_sql_call(tool_call)
        if name == EXECUTE_OPERATION:
            return await self._execute_operation_call(
                tool_call, self._operation_executor, allow_changes, budget
            )
        error = (
            f"Unknown tool '{name}'. Available tools: {', '.join(self._tool_names)}."
        )
        return (
            ToolMessage(
                content=json.dumps({"error": error}),
                tool_call_id=tool_call["id"],
                name=name,
            ),
            False,
            "",
            error,
        )

    async def _execute_sql_call(
        self, tool_call: ToolCall
    ) -> tuple[ToolMessage, bool, str, str]:
        """Run an ``execute_sql`` call; it succeeds when rows come back."""
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
            end = error_str.find(")", start)
            if end != -1:
                error_category = error_str[start:end]
        return (
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=EXECUTE_SQL,
            ),
            succeeded,
            error_category,
            error_str,
        )

    async def _execute_operation_call(
        self,
        tool_call: ToolCall,
        executor: OperationExecutor,
        allow_changes: bool,
        budget: MutationBudget,
    ) -> tuple[ToolMessage, bool, str, str]:
        """Run an ``execute_operation`` call; it succeeds on a final outcome."""
        try:
            call = DataFabricExecuteOperationInput.model_validate(
                tool_call.get("args", {})
            )
        except ValidationError as e:
            result: dict[str, Any] = {"error": f"Invalid arguments: {e}"}
        else:
            # Called directly rather than through the tool, so this run's
            # allow_changes and budget reach it; the model can set neither.
            result = await executor(
                call.entity_name,
                call.operation_name,
                call.arguments,
                allow_changes=allow_changes,
                budget=budget,
            )
        outcome = result.get("outcome")
        succeeded = outcome in OPERATION_SUCCESS_OUTCOMES
        error_str = ""
        if not succeeded:
            error_str = str(
                result.get("error")
                or f"{outcome}: {'; '.join(result.get('errors') or [])}"
            )
        return (
            ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=tool_call["id"],
                name=EXECUTE_OPERATION,
            ),
            succeeded,
            "",
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
            name=EXECUTE_SQL,
            description=(
                f"Execute a SQL SELECT query against Data Fabric entities: {entity_names}. "
                "Refer to the entity schemas in the system message for available "
                "tables and columns. Retry with a corrected query on errors."
            ),
            args_schema=DataFabricExecuteSqlInput,
            coroutine=QueryExecutor(entities_service, entities),
            metadata={"tool_type": "datafabric_sql"},
        )

    def _create_execute_operation_tool(
        self, executor: OperationExecutor, entities: list[Entity]
    ) -> BaseTool:
        """Create the inner ``execute_operation`` tool, bound for its schema.

        Its calls are run by ``_execute_operation_call``. Invoked on its own it
        refuses every Mutation.
        """
        entity_names = ", ".join(e.name for e in entities if entity_operations(e))
        return BaseUiPathStructuredTool(
            name=EXECUTE_OPERATION,
            description=(
                f"Run an operation declared by a Data Fabric entity: {entity_names}. "
                "The operations, their kind and parameters are listed in the system "
                "message. A Mutation changes data and is refused unless the request "
                "allows changes."
            ),
            args_schema=DataFabricExecuteOperationInput,
            coroutine=executor,
            metadata={"tool_type": "datafabric_operation"},
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
