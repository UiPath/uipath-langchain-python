"""Tests for tool_node.py module."""

from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel

from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    UiPathToolNode,
    create_tool_node,
)


class MockTool(BaseTool):
    """Mock tool for testing."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, input_text: str = "") -> str:
        return f"Mock result: {input_text}"

    async def _arun(self, input_text: str = "") -> str:
        return f"Async mock result: {input_text}"


class MockToolWithWrappers(BaseTool, ToolWrapperMixin):
    """Mock tool with wrapper mixin for testing."""

    name: str = "mock_tool_with_wrappers"
    description: str = "A mock tool with wrappers for testing"

    def _run(self, input_text: str = "") -> str:
        return f"Wrapped mock result: {input_text}"

    async def _arun(self, input_text: str = "") -> str:
        return f"Async wrapped mock result: {input_text}"


class FilteredState(BaseModel):
    """Mock filtered state model for testing wrappers."""

    user_id: str = "test_user"
    session_id: str = "test_session"


class MockState(BaseModel):
    """Mock state for testing."""

    messages: list[Any] = []
    user_id: str = "test_user"
    session_id: str = "test_session"


def mock_wrapper(
    tool: BaseTool, call: ToolCall, state: FilteredState
) -> Dict[str, Any]:
    """Mock synchronous wrapper for testing."""
    result = tool.invoke(call["args"])
    # Access user_id if available, demonstrating flexibility with different BaseModel subclasses
    user_id = getattr(state, "user_id", "unknown")
    return {"wrapped_result": f"Wrapped: {result} (user: {user_id})"}


async def mock_awrapper(
    tool: BaseTool, call: ToolCall, state: FilteredState
) -> Dict[str, Any]:
    """Mock asynchronous wrapper for testing."""
    result = await tool.ainvoke(call["args"])
    # Access user_id if available, demonstrating flexibility with different BaseModel subclasses
    user_id = getattr(state, "user_id", "unknown")
    return {"wrapped_result": f"Async wrapped: {result} (user: {user_id})"}


def mock_wrapper_with_command(
    tool: BaseTool, call: ToolCall, state: FilteredState
) -> Command[Any]:
    """Mock wrapper that returns a Command for testing."""
    return Command(goto="next_node")


class TestUiPathToolNode:
    """Test cases for UiPathToolNode class."""

    @pytest.fixture
    def mock_tool(self):
        """Fixture for mock tool."""
        return MockTool()

    @pytest.fixture
    def mock_state(self):
        """Fixture for mock state with tool call."""
        tool_call = {
            "name": "mock_tool",
            "args": {"input_text": "test input"},
            "id": "test_call_id",
        }
        ai_message = AIMessage(content="Using tool", tool_calls=[tool_call])
        return MockState(messages=[ai_message])

    @pytest.fixture
    def empty_state(self):
        """Fixture for state without tool calls."""
        ai_message = AIMessage(content="No tools")
        return MockState(messages=[ai_message])

    @pytest.fixture
    def non_ai_state(self):
        """Fixture for state with non-AI message."""
        human_message = HumanMessage(content="Hello")
        return MockState(messages=[human_message])

    def test_basic_tool_execution(self, mock_tool, mock_state):
        """Test basic tool execution without wrappers."""
        node = UiPathToolNode(mock_tool)

        result = node._func(mock_state)

        assert result is not None
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1

        tool_message = result["messages"][0]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.name == "mock_tool"
        assert tool_message.tool_call_id == "test_call_id"
        assert "Mock result: test input" in tool_message.content

    async def test_async_tool_execution(self, mock_tool, mock_state):
        """Test asynchronous tool execution without wrappers."""
        node = UiPathToolNode(mock_tool)

        result = await node._afunc(mock_state)

        assert result is not None
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1

        tool_message = result["messages"][0]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.name == "mock_tool"
        assert tool_message.tool_call_id == "test_call_id"
        assert "Async mock result: test input" in tool_message.content

    def test_tool_execution_with_sync_wrapper(self, mock_tool, mock_state):
        """Test tool execution with synchronous wrapper."""
        node = UiPathToolNode(mock_tool, wrapper=mock_wrapper)

        result = node._func(mock_state)

        assert result is not None
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1

        tool_message = result["messages"][0]
        assert isinstance(tool_message, ToolMessage)
        assert "wrapped_result" in tool_message.content
        assert "Wrapped:" in tool_message.content
        assert "user: test_user" in tool_message.content

    async def test_tool_execution_with_async_wrapper(self, mock_tool, mock_state):
        """Test tool execution with asynchronous wrapper."""
        node = UiPathToolNode(mock_tool, awrapper=mock_awrapper)

        result = await node._afunc(mock_state)

        assert result is not None
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1

        tool_message = result["messages"][0]
        assert isinstance(tool_message, ToolMessage)
        assert "wrapped_result" in tool_message.content
        assert "Async wrapped:" in tool_message.content
        assert "user: test_user" in tool_message.content

    def test_wrapper_returning_command(self, mock_tool, mock_state):
        """Test wrapper that returns a Command instead of a dict."""
        node = UiPathToolNode(mock_tool, wrapper=mock_wrapper_with_command)

        result = node._func(mock_state)

        assert isinstance(result, Command)
        assert result.goto == "next_node"

    def test_no_tool_calls_returns_none(self, mock_tool, empty_state):
        """Test that missing tool calls return None."""
        node = UiPathToolNode(mock_tool)

        result = node._func(empty_state)

        assert result is None

    def test_non_ai_message_raises_error(self, mock_tool, non_ai_state):
        """Test that non-AI messages raise ValueError."""
        node = UiPathToolNode(mock_tool)

        with pytest.raises(
            ValueError, match="Last message in message stack is not an AIMessage"
        ):
            node._func(non_ai_state)

    def test_mismatched_tool_name_returns_none(self, mock_tool, mock_state):
        """Test that mismatched tool names return None."""
        # Change the tool call name to something different
        mock_state.messages[-1].tool_calls[0]["name"] = "different_tool"

        node = UiPathToolNode(mock_tool)

        result = node._func(mock_state)

        assert result is None

    def test_state_filtering(self, mock_tool, mock_state):
        """Test that state is properly filtered for wrapper functions."""
        node = UiPathToolNode(mock_tool, wrapper=mock_wrapper)

        filtered_state = node._filter_state(mock_state, mock_wrapper)

        assert isinstance(filtered_state, FilteredState)
        assert filtered_state.user_id == "test_user"
        assert filtered_state.session_id == "test_session"

    def test_invalid_wrapper_state_type_raises_error(self, mock_tool, mock_state):
        """Test that invalid wrapper state parameter types raise ValueError."""

        def invalid_wrapper(
            tool: BaseTool, call: ToolCall, state: str
        ) -> Dict[str, Any]:
            return {"result": "invalid"}

        node = UiPathToolNode(mock_tool, wrapper=invalid_wrapper)

        with pytest.raises(
            ValueError,
            match="Wrapper state parameter must be a pydantic BaseModel subclass",
        ):
            node._func(mock_state)


class TestToolWrapperMixin:
    """Test cases for ToolWrapperMixin class."""

    def test_set_tool_wrappers(self):
        """Test setting tool wrappers on a mixin instance."""
        tool = MockToolWithWrappers()

        tool.set_tool_wrappers(wrapper=mock_wrapper, awrapper=mock_awrapper)

        assert tool.wrapper == mock_wrapper
        assert tool.awrapper == mock_awrapper

    def test_set_tool_wrappers_with_none(self):
        """Test setting tool wrappers to None."""
        tool = MockToolWithWrappers()

        tool.set_tool_wrappers(wrapper=None, awrapper=None)

        assert tool.wrapper is None
        assert tool.awrapper is None


class TestCreateToolNode:
    """Test cases for create_tool_node function."""

    def test_create_tool_node_basic_tools(self):
        """Test creating tool nodes for basic tools."""
        tools = [MockTool(name="mock_tool_1"), MockTool(name="mock_tool_2")]

        result = create_tool_node(tools)

        assert len(result) == 2
        assert "mock_tool_1" in result
        assert isinstance(result["mock_tool_1"], UiPathToolNode)
        assert result["mock_tool_1"].tool.name == "mock_tool_1"

        assert "mock_tool_2" in result
        assert isinstance(result["mock_tool_2"], UiPathToolNode)
        assert result["mock_tool_2"].tool.name == "mock_tool_2"

    def test_create_tool_node_with_wrapper_mixin(self):
        """Test creating tool nodes for tools with wrapper mixin."""
        tool_with_wrappers = MockToolWithWrappers()
        tool_with_wrappers.set_tool_wrappers(
            wrapper=mock_wrapper, awrapper=mock_awrapper
        )

        tools = [MockTool(), tool_with_wrappers]

        result = create_tool_node(tools)

        assert len(result) == 2
        assert "mock_tool" in result
        assert "mock_tool_with_wrappers" in result

        # Basic tool should have no wrappers
        basic_node = result["mock_tool"]
        assert basic_node.wrapper is None
        assert basic_node.awrapper is None

        # Tool with mixin should have wrappers
        wrapper_node = result["mock_tool_with_wrappers"]
        assert wrapper_node.wrapper == mock_wrapper
        assert wrapper_node.awrapper == mock_awrapper

    def test_create_tool_node_empty_tools(self):
        """Test creating tool nodes with empty tool list."""
        result = create_tool_node([])

        assert result == {}
