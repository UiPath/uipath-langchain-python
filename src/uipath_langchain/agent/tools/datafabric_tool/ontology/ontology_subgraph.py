"""Inner LangGraph sub-graph for the ontology tool.

A self-contained ReAct loop that turns a natural-language question into SQL over
the ontology's business objects, runs it through the ``execute_sql`` tool bound to
the Ontology service, retries on a rejected statement and short-circuits straight
to END on a successful execution so the outer agent receives the raw rows.

The sub-graph is prompt-agnostic: the caller passes the ready system prompt (see
:mod:`ontology_service_prompt`) and the ``execute_sql`` tool (see
:mod:`ontology_service_sql`). It shares nothing with the entity-set tool's
``datafabric_subgraph`` beyond the pattern.
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

logger = logging.getLogger(__name__)


class OntologySubgraphState(BaseModel):
    """State for the inner ontology ReAct sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    iteration_count: int = 0
    last_tool_success: bool = False


class OntologyGraph:
    """Inner ReAct sub-graph: the model writes SQL, ``execute_sql`` runs it.

    Each graph node is a method. The graph is compiled during ``__init__`` and is
    available as ``compiled_graph``.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        execute_sql_tool: BaseTool,
        max_iterations: int = 25,
    ) -> None:
        self._max_iterations = max_iterations
        self._execute_sql_tool = execute_sql_tool
        self._system_message = SystemMessage(content=system_prompt)
        self._inner_llm = llm.model_copy(update={"disable_streaming": True}).bind_tools(
            [execute_sql_tool]
        )

        graph = StateGraph(OntologySubgraphState)
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

    async def llm_node(self, state: OntologySubgraphState) -> dict[str, Any]:
        """Invoke the inner LLM with the current message history."""
        messages = [self._system_message] + list(state.messages)
        response = await self._inner_llm.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(self, state: OntologySubgraphState) -> dict[str, Any]:
        """Execute all tool calls from the last AIMessage concurrently."""
        last = state.messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"iteration_count": state.iteration_count}

        results = await asyncio.gather(
            *[self._execute_tool_call(tc) for tc in last.tool_calls]
        )
        tool_messages = [msg for msg, _ in results]
        all_succeeded = bool(results) and all(success for _, success in results)
        return {
            "messages": tool_messages,
            "iteration_count": state.iteration_count + len(last.tool_calls),
            "last_tool_success": all_succeeded,
        }

    async def _execute_tool_call(self, tool_call: ToolCall) -> tuple[ToolMessage, bool]:
        """Execute a single tool call and report whether it succeeded."""
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
        succeeded = (
            isinstance(result, dict)
            and not result.get("error")
            and result.get("total_count", 0) > 0
        )
        return (
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name="execute_sql",
            ),
            succeeded,
        )

    async def termination_node(self, state: OntologySubgraphState) -> dict[str, Any]:
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

    def router(self, state: OntologySubgraphState) -> str:
        """Route from ``inner_llm`` to tool, termination, or END."""
        last = state.messages[-1] if state.messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            if state.iteration_count < self._max_iterations:
                return "inner_tool"
            return "termination"
        return END

    def tool_router(self, state: OntologySubgraphState) -> str:
        """Route from ``inner_tool``: short-circuit on success, retry on error.

        A successful execution ends the sub-graph so the outer agent reads the raw
        rows; a rejected statement loops back to ``inner_llm`` with the runtime's
        message so the model can correct it, within ``max_iterations``.
        """
        if state.last_tool_success:
            return END
        return "inner_llm"

    @staticmethod
    def create(
        llm: BaseChatModel,
        system_prompt: str,
        execute_sql_tool: BaseTool,
        max_iterations: int = 25,
    ) -> CompiledStateGraph[Any]:
        """Create and return a compiled ontology sub-graph."""
        return OntologyGraph(
            llm, system_prompt, execute_sql_tool, max_iterations
        ).compiled_graph
