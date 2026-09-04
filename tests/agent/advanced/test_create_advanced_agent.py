"""Tests for create_advanced_agent."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from uipath_langchain.agent.advanced.agent import create_advanced_agent


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


class TestCreateAdvancedAgent:
    """Test the create_advanced_agent function."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        return _make_mock_model()

    def test_advanced_agent_with_tools(self, mock_model: MagicMock) -> None:
        """Custom tool is registered alongside built-in advanced agent tools."""
        result = create_advanced_agent(
            mock_model, system_prompt="test", tools=[_sample_tool]
        )
        assert isinstance(result, CompiledStateGraph)
        tools_node = result.nodes["tools"].bound
        assert isinstance(tools_node, ToolNode)
        tool_names = set(tools_node.tools_by_name.keys())
        assert "_sample_tool" in tool_names

    def test_advanced_agent_without_tools(self, mock_model: MagicMock) -> None:
        """Built-in advanced agent tools are present even with no custom tools."""
        result = create_advanced_agent(mock_model, system_prompt="test", tools=[])
        assert isinstance(result, CompiledStateGraph)
        tools_node = result.nodes["tools"].bound
        assert isinstance(tools_node, ToolNode)
        tool_names = set(tools_node.tools_by_name.keys())
        assert "write_todos" in tool_names

    def test_advanced_agent_converts_sequences_to_lists(
        self, mock_model: MagicMock
    ) -> None:
        """Tuples for tools and subagents are converted to lists."""
        with patch(
            "uipath_langchain.agent.advanced.agent._create_deep_agent"
        ) as mock_upstream:
            mock_upstream.return_value = MagicMock(spec=CompiledStateGraph)
            create_advanced_agent(
                mock_model,
                system_prompt="test",
                tools=(_sample_tool,),
                subagents=(),
            )
            _, kwargs = mock_upstream.call_args
            assert isinstance(kwargs["tools"], list)
            assert isinstance(kwargs["subagents"], list)

    def test_advanced_agent_forwards_skills(self, mock_model: MagicMock) -> None:
        """Non-empty skills are forwarded to the upstream builder as a list."""
        with patch(
            "uipath_langchain.agent.advanced.agent._create_deep_agent"
        ) as mock_upstream:
            mock_upstream.return_value = MagicMock(spec=CompiledStateGraph)
            create_advanced_agent(
                mock_model, system_prompt="test", skills=("/skills/",)
            )
            _, kwargs = mock_upstream.call_args
            assert kwargs["skills"] == ["/skills/"]

    def test_advanced_agent_empty_skills_becomes_none(
        self, mock_model: MagicMock
    ) -> None:
        """An empty skills sequence collapses to None (disables the middleware)."""
        with patch(
            "uipath_langchain.agent.advanced.agent._create_deep_agent"
        ) as mock_upstream:
            mock_upstream.return_value = MagicMock(spec=CompiledStateGraph)
            create_advanced_agent(mock_model, system_prompt="test")
            _, kwargs = mock_upstream.call_args
            assert kwargs["skills"] is None
