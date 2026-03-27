"""Inner LangGraph sub-graph for the Data Fabric agentic tool.

Implements a self-contained ReAct loop where an inner LLM translates
natural-language questions into SQL, executes them via ``execute_sql``,
and retries on errors — all within a single outer tool call.
"""

import logging
from typing import Annotated, Any

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
from pydantic import BaseModel
from uipath.platform.entities import Entity

from ..base_uipath_structured_tool import BaseUiPathStructuredTool
from .models import DataFabricExecuteSqlInput
from .schema_context import format_schemas_for_context

logger = logging.getLogger(__name__)

_MAX_RECORDS_IN_RESPONSE = 50


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DataFabricSubgraphState(BaseModel):
    """State for the inner Data Fabric ReAct sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    iteration_count: int = 0


# ---------------------------------------------------------------------------
# Inner tool factory
# ---------------------------------------------------------------------------


def _create_execute_sql_tool(
    routing_context: Any | None = None,
) -> BaseTool:
    """Create the inner ``execute_sql`` tool used by the sub-graph.

    Args:
        routing_context: Optional routing context for entity queries.
            Can be ``None`` for backward compatibility with older SDK versions.

    Returns:
        A ``BaseTool`` that executes SQL against Data Fabric entities.
    """

    async def _execute_sql(sql_query: str) -> dict[str, Any]:
        from uipath.platform import UiPath

        logger.debug("execute_sql called with SQL: %s", sql_query)

        sdk = UiPath()
        try:
            kwargs: dict[str, Any] = {"sql_query": sql_query}
            if routing_context is not None:
                kwargs["routing_context"] = routing_context

            records = await sdk.entities.query_entity_records_async(**kwargs)
            total_count = len(records)
            truncated = total_count > _MAX_RECORDS_IN_RESPONSE
            returned_records = (
                records[:_MAX_RECORDS_IN_RESPONSE] if truncated else records
            )

            result: dict[str, Any] = {
                "records": returned_records,
                "total_count": total_count,
                "returned_count": len(returned_records),
                "sql_query": sql_query,
            }
            if truncated:
                result["truncated"] = True
                result["message"] = (
                    f"Showing {len(returned_records)} of {total_count} records. "
                    "Use more specific filters or LIMIT to narrow results."
                )
            return result
        except Exception as e:
            logger.error("SQL query failed: %s", e)
            return {
                "records": [],
                "total_count": 0,
                "error": str(e),
                "sql_query": sql_query,
            }

    return BaseUiPathStructuredTool(
        name="execute_sql",
        description=(
            "Execute a SQL SELECT query against Data Fabric entities. "
            "Refer to the entity schemas in the system message for available "
            "tables and columns. Retry with a corrected query on errors."
        ),
        args_schema=DataFabricExecuteSqlInput,
        coroutine=_execute_sql,
        metadata={"tool_type": "datafabric_sql"},
    )


# ---------------------------------------------------------------------------
# System message builder
# ---------------------------------------------------------------------------


def build_inner_system_message(entities: list[Entity]) -> SystemMessage:
    """Build the system message for the inner sub-graph LLM.

    Args:
        entities: List of Entity objects whose schemas should be included.

    Returns:
        A ``SystemMessage`` instructing the LLM to translate NL to SQL.
    """
    preamble = (
        "You are a SQL assistant. Your job is to translate the user's "
        "natural-language question into a SQL query and execute it using "
        "the `execute_sql` tool.\n\n"
        "Rules:\n"
        "- Always call `execute_sql` with a valid SQL SELECT statement.\n"
        "- If the query returns an error, analyse the error message and "
        "retry with a corrected query.\n"
        "- Once you have the result, return the final answer to the user.\n"
        "- Do NOT fabricate data — only use values returned by `execute_sql`.\n\n"
    )

    schema_context = format_schemas_for_context(entities)
    content = preamble + schema_context

    return SystemMessage(content=content)


# ---------------------------------------------------------------------------
# Sub-graph factory
# ---------------------------------------------------------------------------


def create_datafabric_subgraph(
    llm: Any,
    entities: list[Entity],
    routing_context: Any | None = None,
    max_iterations: int = 5,
) -> Any:
    """Create a compiled LangGraph sub-graph for Data Fabric SQL execution.

    The graph implements a simple ReAct loop:

    .. code-block:: text

        START -> inner_llm -> (tool_calls?) -> inner_tool -> inner_llm -> ... -> END

    Args:
        llm: The chat model to use inside the sub-graph.
        entities: Entity objects whose schemas should be provided as context.
        routing_context: Optional routing context forwarded to the SDK query.
        max_iterations: Maximum number of tool-call iterations before forcing
            the graph to terminate.

    Returns:
        A compiled ``StateGraph`` ready to be invoked with
        ``DataFabricSubgraphState``.
    """
    # --- inner tool ---
    execute_sql_tool = _create_execute_sql_tool(routing_context)

    # --- system message ---
    system_message = build_inner_system_message(entities)

    # --- inner LLM ---
    inner_llm = llm.model_copy(update={"disable_streaming": True}).bind_tools(
        [execute_sql_tool], parallel_tool_calls=False
    )

    async def inner_llm_node(
        state: DataFabricSubgraphState,
    ) -> dict[str, Any]:
        """Invoke the inner LLM with the current message history."""
        messages = [system_message] + list(state.messages)
        response = await inner_llm.ainvoke(messages)
        return {"messages": [response]}

    # --- inner tool node ---
    async def inner_tool_node(
        state: DataFabricSubgraphState,
    ) -> dict[str, Any]:
        """Extract tool call from the last AIMessage, invoke execute_sql."""
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"iteration_count": state.iteration_count}

        tool_call = last_message.tool_calls[0]
        tool_input = tool_call.get("args", {})
        tool_result = await execute_sql_tool.ainvoke(tool_input)

        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"],
            name="execute_sql",
        )
        return {
            "messages": [tool_message],
            "iteration_count": state.iteration_count + 1,
        }

    # --- router ---
    def router(state: DataFabricSubgraphState) -> str:
        """Route to inner_tool if the LLM requested a tool call and we
        have not exceeded max_iterations; otherwise route to END."""
        last_message = state.messages[-1] if state.messages else None
        if (
            isinstance(last_message, AIMessage)
            and last_message.tool_calls
            and state.iteration_count < max_iterations
        ):
            return "inner_tool"
        return END

    # --- build graph ---
    graph = StateGraph(DataFabricSubgraphState)

    graph.add_node("inner_llm", inner_llm_node)
    graph.add_node("inner_tool", inner_tool_node)

    graph.add_edge(START, "inner_llm")
    graph.add_conditional_edges("inner_llm", router, ["inner_tool", END])
    graph.add_edge("inner_tool", "inner_llm")

    return graph.compile()
