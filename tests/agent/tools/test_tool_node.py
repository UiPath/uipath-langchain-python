"""Tests for tool node factory."""

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from uipath_langchain.agent.tools.tool_node import create_tool_node


class TestCreateToolNode:
    """Test cases for create_tool_node function."""

    def test_returns_dict(self):
        """Should return a dictionary."""

        @tool
        def test_tool(x: str) -> str:
            """A test tool."""
            return x

        result = create_tool_node([test_tool])
        assert isinstance(result, dict)

    def test_maps_tool_name_to_node(self):
        """Should create mapping from tool name to ToolNode."""

        @tool
        def my_tool(x: str) -> str:
            """A test tool."""
            return x

        result = create_tool_node([my_tool])
        assert "my_tool" in result

    def test_each_tool_gets_own_node(self):
        """Should create separate ToolNode for each tool."""

        @tool
        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        @tool
        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        result = create_tool_node([tool_a, tool_b])
        assert len(result) == 2
        assert "tool_a" in result
        assert "tool_b" in result

    def test_nodes_are_tool_node_instances(self):
        """Should create ToolNode instances."""

        @tool
        def test_tool(x: str) -> str:
            """A test tool."""
            return x

        result = create_tool_node([test_tool])
        assert isinstance(result["test_tool"], ToolNode)

    def test_empty_tools_list(self):
        """Should return empty dict for empty tools list."""
        result = create_tool_node([])
        assert result == {}

    def test_preserves_tool_names(self):
        """Should use exact tool names as keys."""

        @tool("special-tool_123")
        def special_tool(x: str) -> str:
            """A test tool."""
            return x

        result = create_tool_node([special_tool])
        assert "special-tool_123" in result
