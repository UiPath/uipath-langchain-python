"""Inner LangGraph sub-graph for the Data Fabric agentic tool.

Implements a self-contained ReAct loop where an inner LLM translates
natural-language questions into SQL, executes them via ``execute_sql``,
and retries on errors — all within a single outer tool call.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from uipath.platform.entities import Entity, QueryRoutingOverrideContext

from ..sql_tool import SqlTool
from . import prompt_builder
from .models import DataFabricExecuteSqlInput

logger = logging.getLogger(__name__)

_DEBUG_FILE = "/tmp/df_subgraph_debug.txt"


def _debug_log(section: str, data: Any) -> None:
    """Append debug info to /tmp/df_subgraph_debug.txt."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(_DEBUG_FILE, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"[{timestamp}] {section}\n")
            f.write(f"{'=' * 80}\n")
            if isinstance(data, str):
                f.write(data)
            else:
                f.write(json.dumps(data, indent=2, default=str))
            f.write("\n")
    except Exception:
        pass


# --- State ---
class DataFabricSubgraphState(BaseModel):
    """State for the inner Data Fabric ReAct sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    iteration_count: int = 0


# --- SQL Executor ---
class SqlExecutor:
    """Executes SQL queries against Data Fabric."""

    def __init__(self, routing_context: QueryRoutingOverrideContext) -> None:
        from uipath.platform import UiPath

        self._sdk = UiPath()
        self._routing_context = routing_context

    async def __call__(self, sql_query: str) -> dict[str, Any]:
        logger.debug("execute_sql called with SQL: %s", sql_query)
        try:
            records = await self._sdk.entities.query_entity_records_async(
                sql_query=sql_query,
                routing_context=self._routing_context,
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


def _create_execute_sql_tool(
    routing_context: QueryRoutingOverrideContext,
    entities: list[Entity],
) -> BaseTool:
    """Create the inner ``execute_sql`` tool for the sub-graph."""
    entity_names = ", ".join(e.name for e in entities)
    return SqlTool(
        name="execute_sql",
        description=(
            f"Execute a SQL SELECT query against Data Fabric entities: {entity_names}. "
            "Refer to the entity schemas in the system message for available "
            "tables and columns. Retry with a corrected query on errors."
        ),
        args_schema=DataFabricExecuteSqlInput,
        coroutine=SqlExecutor(routing_context),
        metadata={"tool_type": "datafabric_sql"},
    )


# --- System message builder ---
def build_inner_system_message(
    entities: list[Entity],
    resource_description: str = "",
) -> SystemMessage:
    """Build the system message for the inner sub-graph LLM.

    Args:
        entities: List of Entity objects whose schemas should be included.
        resource_description: Optional description of the resource/entity set.
    """
    return SystemMessage(
        content=prompt_builder.build(entities, resource_description)
    )


# --- Sub-graph builder ---
class DataFabricGraphBuilder:
    """Builds and compiles the inner ReAct sub-graph.

    Each graph node is a method — no closures inside closures.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        entities: list[Entity],
        routing_context: QueryRoutingOverrideContext,
        max_iterations: int = 5,
        resource_description: str = "",
    ) -> None:
        self._max_iterations = max_iterations
        self._execute_sql_tool = _create_execute_sql_tool(routing_context, entities)
        self._system_message = build_inner_system_message(entities, resource_description)
        self._inner_llm = llm.model_copy(update={"disable_streaming": True}).bind_tools(
            [self._execute_sql_tool]
        )
        _debug_log("INIT — Tool description", {
            "tool_name": self._execute_sql_tool.name,
            "tool_description": self._execute_sql_tool.description,
            "tool_args_schema": self._execute_sql_tool.args_schema.model_json_schema()
            if self._execute_sql_tool.args_schema else None,
        })
        _debug_log("INIT — System message", self._system_message.content)

    async def llm_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Invoke the inner LLM with the current message history."""
        messages = [self._system_message] + list(state.messages)
        _debug_log(f"LLM_NODE — Input (iteration={state.iteration_count})", [
            {"role": type(m).__name__, "content": str(m.content),
             **({"tool_calls": m.tool_calls} if hasattr(m, "tool_calls") and m.tool_calls else {})}
            for m in messages
        ])
        response = await self._inner_llm.ainvoke(messages)
        _debug_log("LLM_NODE — Response", {
            "content": str(response.content) if response.content else None,
            "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else None,
        })
        return {"messages": [response]}

    async def tool_node(self, state: DataFabricSubgraphState) -> dict[str, Any]:
        """Execute all tool calls from the last AIMessage concurrently."""
        last = state.messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"iteration_count": state.iteration_count}

        tool_messages = await asyncio.gather(
            *[self._execute_tool_call(tc) for tc in last.tool_calls]
        )
        return {
            "messages": list(tool_messages),
            "iteration_count": state.iteration_count + len(last.tool_calls),
        }

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> ToolMessage:
        """Execute a single tool call and wrap the result."""
        args = tool_call.get("args", {})
        _debug_log("TOOL_CALL — Input", {
            "tool_call_id": tool_call.get("id"),
            "args": args,
        })
        try:
            result = await self._execute_sql_tool.ainvoke(args)
        except ValueError as e:
            result = {
                "records": [],
                "total_count": 0,
                "error": str(e),
                "sql_query": args.get("sql_query", ""),
            }
        _debug_log("TOOL_CALL — Result", result)
        return ToolMessage(
            content=str(result),
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

    def compile(self) -> CompiledStateGraph:
        """Build and compile the StateGraph."""
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

        return graph.compile()


def create_datafabric_subgraph(
    llm: BaseChatModel,
    entities: list[Entity],
    routing_context: QueryRoutingOverrideContext,
    max_iterations: int = 5,
    resource_description: str = "",
) -> CompiledStateGraph:
    """Create a compiled LangGraph sub-graph for Data Fabric SQL execution.

    Args:
        llm: The chat model to use inside the sub-graph.
        entities: Entity objects whose schemas should be provided as context.
        routing_context: Routing context forwarded to the SDK query.
        max_iterations: Maximum tool-call iterations before forced termination.
        resource_description: Optional description of the resource/entity set.

    Returns:
        A compiled ``StateGraph`` ready to be invoked.
    """
    builder = DataFabricGraphBuilder(
        llm, entities, routing_context, max_iterations, resource_description
    )
    return builder.compile()
