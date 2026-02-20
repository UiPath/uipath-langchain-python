"""Tests for create_deep_agent."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from uipath_langchain.agent.deep.agent import create_deep_agent


def _make_mock_model() -> MagicMock:
    """Create a mock model with the attributes the upstream code reads."""
    model = MagicMock(spec=BaseChatModel)
    # upstream reads model.profile for summarization defaults
    model.profile = None
    return model


@tool
def _sample_tool(query: str) -> str:
    """A sample tool for testing."""
    return query


class TestCreateDeepAgent:
    """Test the create_deep_agent function."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        return _make_mock_model()

    def test_deep_agent_with_tools(self, mock_model: MagicMock) -> None:
        """Custom tool is registered alongside built-in deep agent tools."""
        result = create_deep_agent(
            mock_model, system_prompt="test", tools=[_sample_tool]
        )
        assert isinstance(result, CompiledStateGraph)
        tools_node = result.nodes["tools"].bound
        assert isinstance(tools_node, ToolNode)
        tool_names = set(tools_node.tools_by_name.keys())
        assert "_sample_tool" in tool_names

    def test_deep_agent_without_tools(self, mock_model: MagicMock) -> None:
        """Built-in deep agent tools are present even with no custom tools."""
        result = create_deep_agent(mock_model, system_prompt="test", tools=[])
        assert isinstance(result, CompiledStateGraph)
        tools_node = result.nodes["tools"].bound
        assert isinstance(tools_node, ToolNode)
        tool_names = set(tools_node.tools_by_name.keys())
        assert "write_todos" in tool_names

    def test_deep_agent_with_backend(self, mock_model: MagicMock) -> None:
        """Compiles without error when a FilesystemBackend is provided."""
        from uipath_langchain.agent.deep import FilesystemBackend

        with tempfile.TemporaryDirectory() as tmp:
            backend = FilesystemBackend(root_dir=tmp)
            result = create_deep_agent(
                mock_model, system_prompt="test", backend=backend
            )
            assert isinstance(result, CompiledStateGraph)

    def test_deep_agent_converts_sequences_to_lists(
        self, mock_model: MagicMock
    ) -> None:
        """Tuples for tools and subagents are converted to lists."""
        with patch(
            "uipath_langchain.agent.deep.agent._create_deep_agent"
        ) as mock_upstream:
            mock_upstream.return_value = MagicMock(spec=CompiledStateGraph)
            create_deep_agent(
                mock_model,
                system_prompt="test",
                tools=(_sample_tool,),
                subagents=(),
            )
            _, kwargs = mock_upstream.call_args
            assert isinstance(kwargs["tools"], list)
            assert isinstance(kwargs["subagents"], list)
