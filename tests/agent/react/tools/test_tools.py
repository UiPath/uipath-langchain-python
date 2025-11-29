"""Tests for flow control tools."""

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from uipath_langchain.agent.react.tools.tools import (
    create_end_execution_tool,
    create_flow_control_tools,
    create_raise_error_tool,
)


class TestCreateEndExecutionTool:
    """Test cases for create_end_execution_tool function."""

    def test_returns_structured_tool(self):
        """Should return a StructuredTool instance."""
        tool = create_end_execution_tool()
        assert isinstance(tool, StructuredTool)

    def test_uses_default_name(self):
        """Should use end_execution tool name."""
        tool = create_end_execution_tool()
        assert tool.name == END_EXECUTION_TOOL.name

    def test_uses_default_description(self):
        """Should use end_execution tool description."""
        tool = create_end_execution_tool()
        assert tool.description == END_EXECUTION_TOOL.description

    def test_uses_default_schema_when_none_provided(self):
        """Should use END_EXECUTION_TOOL args_schema when no schema provided."""
        tool = create_end_execution_tool()
        assert tool.args_schema is END_EXECUTION_TOOL.args_schema

    def test_uses_custom_output_schema(self):
        """Should use custom schema when provided."""

        class CustomSchema(BaseModel):
            result: str
            code: int

        tool = create_end_execution_tool(agent_output_schema=CustomSchema)
        assert tool.args_schema is CustomSchema

    @pytest.mark.asyncio
    async def test_coroutine_returns_kwargs(self):
        """Should return kwargs passed to tool."""
        tool = create_end_execution_tool()
        result = await tool.coroutine(final_answer="test result")
        assert result == {"final_answer": "test result"}


class TestCreateRaiseErrorTool:
    """Test cases for create_raise_error_tool function."""

    def test_returns_structured_tool(self):
        """Should return a StructuredTool instance."""
        tool = create_raise_error_tool()
        assert isinstance(tool, StructuredTool)

    def test_uses_raise_error_name(self):
        """Should use raise_error tool name."""
        tool = create_raise_error_tool()
        assert tool.name == RAISE_ERROR_TOOL.name

    def test_uses_raise_error_description(self):
        """Should use raise_error tool description."""
        tool = create_raise_error_tool()
        assert tool.description == RAISE_ERROR_TOOL.description

    def test_uses_raise_error_schema(self):
        """Should use RAISE_ERROR_TOOL args_schema."""
        tool = create_raise_error_tool()
        assert tool.args_schema is RAISE_ERROR_TOOL.args_schema

    @pytest.mark.asyncio
    async def test_coroutine_returns_kwargs(self):
        """Should return kwargs passed to tool."""
        tool = create_raise_error_tool()
        result = await tool.coroutine(message="error", details="details")
        assert result == {"message": "error", "details": "details"}


class TestCreateFlowControlTools:
    """Test cases for create_flow_control_tools function."""

    def test_returns_list_of_tools(self):
        """Should return a list of tools."""
        tools = create_flow_control_tools()
        assert isinstance(tools, list)
        assert len(tools) == 2

    def test_contains_end_execution_tool(self):
        """Should contain end_execution tool."""
        tools = create_flow_control_tools()
        tool_names = [t.name for t in tools]
        assert END_EXECUTION_TOOL.name in tool_names

    def test_contains_raise_error_tool(self):
        """Should contain raise_error tool."""
        tools = create_flow_control_tools()
        tool_names = [t.name for t in tools]
        assert RAISE_ERROR_TOOL.name in tool_names

    def test_passes_output_schema_to_end_execution(self):
        """Should pass custom schema to end_execution tool."""

        class CustomSchema(BaseModel):
            output: str

        tools = create_flow_control_tools(agent_output_schema=CustomSchema)
        end_tool = next(t for t in tools if t.name == END_EXECUTION_TOOL.name)
        assert end_tool.args_schema is CustomSchema

    def test_all_tools_are_structured_tools(self):
        """Should return all tools as StructuredTool instances."""
        tools = create_flow_control_tools()
        for tool in tools:
            assert isinstance(tool, StructuredTool)
