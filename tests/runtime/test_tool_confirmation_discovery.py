"""Tests that _get_tool_confirmation_info discovers confirmation tools through ConversationalToolRunnableCallable wrappers.

This is the integration guard against silent regressions: if LangGraph changes its
compiled-graph node structure, or if a new wrapping layer forgets to preserve the
BaseTool reference, these tests will fail.
"""

from typing import Any

from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from uipath_langchain.agent.tools.tool_node import (
    UiPathToolNode,
    wrap_tools_with_error_handling,
)
from uipath_langchain.chat.hitl import REQUIRE_CONVERSATIONAL_CONFIRMATION
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime


class _ConfirmableInput(BaseModel):
    query: str = Field(description="The query to confirm")


class _ConfirmableTool(BaseTool):
    name: str = "needs_confirmation"
    description: str = "A tool that requires user confirmation"
    args_schema: type[BaseModel] = _ConfirmableInput
    metadata: dict[str, Any] = {REQUIRE_CONVERSATIONAL_CONFIRMATION: True}

    def _run(self, query: str) -> str:
        return f"confirmed: {query}"


class _NormalTool(BaseTool):
    name: str = "no_confirmation"
    description: str = "A normal tool"

    def _run(self) -> str:
        return "done"


class _MinimalState(BaseModel):
    value: str = ""


def _compile_graph_with_wrapped_tools(tools: list[BaseTool]):
    """Build and compile a minimal graph with tools wrapped through the standard pipeline."""
    tool_nodes = {t.name: UiPathToolNode(t) for t in tools}
    wrapped = wrap_tools_with_error_handling(tool_nodes)

    builder: StateGraph[_MinimalState] = StateGraph(_MinimalState)
    names = list(wrapped.keys())
    for name, node in wrapped.items():
        builder.add_node(name, node)

    # Wire START → first tool → END (graph must be connected to compile)
    builder.add_edge(START, names[0])
    for i in range(len(names) - 1):
        builder.add_edge(names[i], names[i + 1])
    builder.add_edge(names[-1], END)

    return builder.compile()


class TestToolConfirmationDiscovery:
    def test_discovers_confirmation_tool_through_wrapper(self):
        graph = _compile_graph_with_wrapped_tools([_ConfirmableTool(), _NormalTool()])
        runtime = UiPathLangGraphRuntime(graph)

        schemas = runtime.chat.tool_confirmation_schemas
        assert "needs_confirmation" in schemas
        assert "no_confirmation" not in schemas

    def test_schema_contains_input_properties(self):
        graph = _compile_graph_with_wrapped_tools([_ConfirmableTool()])
        runtime = UiPathLangGraphRuntime(graph)

        schema = runtime.chat.tool_confirmation_schemas["needs_confirmation"]
        assert "properties" in schema
        assert "query" in schema["properties"]

    def test_empty_when_no_confirmation_tools(self):
        graph = _compile_graph_with_wrapped_tools([_NormalTool()])
        runtime = UiPathLangGraphRuntime(graph)

        assert runtime.chat.tool_confirmation_schemas == {}
