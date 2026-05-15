"""Tests that _get_client_side_tools discovers client-side tools through RunnableCallableWithTool wrappers.

Integration guard: if the wrapping pipeline changes and stops preserving the
BaseTool reference for client-side tools, these tests will fail.
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
from uipath_langchain.chat.hitl import IS_CONVERSATIONAL_CLIENT_SIDE_TOOL
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime


class _ClientSideInput(BaseModel):
    title: str = Field(description="Movie title")


class _ClientSideTool(BaseTool):
    name: str = "client_tool"
    description: str = "A client-side tool"
    args_schema: type[BaseModel] = _ClientSideInput
    metadata: dict[str, Any] = {
        IS_CONVERSATIONAL_CLIENT_SIDE_TOOL: True,
        "output_schema": {
            "type": "object",
            "properties": {"rating": {"type": "number"}},
        },
    }

    def _run(self, title: str) -> str:
        return f"result for {title}"


class _NormalTool(BaseTool):
    name: str = "normal_tool"
    description: str = "A normal server tool"

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

    builder.add_edge(START, names[0])
    for i in range(len(names) - 1):
        builder.add_edge(names[i], names[i + 1])
    builder.add_edge(names[-1], END)

    return builder.compile()


class TestClientSideToolDiscovery:
    def test_discovers_client_side_tool_through_wrapper(self):
        graph = _compile_graph_with_wrapped_tools([_ClientSideTool(), _NormalTool()])
        runtime = UiPathLangGraphRuntime(graph)

        client_tools = runtime.chat.client_side_tools
        assert "client_tool" in client_tools
        assert "normal_tool" not in client_tools

    def test_schemas_are_preserved(self):
        graph = _compile_graph_with_wrapped_tools([_ClientSideTool()])
        runtime = UiPathLangGraphRuntime(graph)

        tool_info = runtime.chat.client_side_tools["client_tool"]
        assert tool_info is not None
        assert "output_schema" in tool_info
        assert "input_schema" in tool_info
        assert "rating" in tool_info["output_schema"]["properties"]
        assert "title" in tool_info["input_schema"]["properties"]

    def test_empty_when_no_client_side_tools(self):
        graph = _compile_graph_with_wrapped_tools([_NormalTool()])
        runtime = UiPathLangGraphRuntime(graph)

        assert runtime.chat.client_side_tools == {}
