"""Tests for create_advanced_agent."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from uipath.agent.models.agent import AgentInternalToolResourceConfig

from uipath_langchain.agent.advanced.agent import create_advanced_agent
from uipath_langchain.agent.tools.internal_tools.create_file_tool import (
    CreateFileTool,
    create_file_tool,
)


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
        """Built-in filesystem tools are present even with no custom tools."""
        result = create_advanced_agent(mock_model, system_prompt="test", tools=[])
        assert isinstance(result, CompiledStateGraph)
        tools_node = result.nodes["tools"].bound
        assert isinstance(tools_node, ToolNode)
        tool_names = set(tools_node.tools_by_name.keys())
        assert {"ls", "read_file", "write_file"} <= tool_names

    def test_advanced_agent_has_no_todo_tool(self, mock_model: MagicMock) -> None:
        """``write_todos`` is deliberately absent.

        deepagents 0.7.0 dropped ``TodoListMiddleware`` from its defaults on
        benchmark evidence (langchain-ai/deepagents#4929) and we do not restore it.
        This pins that decision so a future change has to be deliberate.
        """
        result = create_advanced_agent(mock_model, system_prompt="test", tools=[])
        tools_node = result.nodes["tools"].bound
        assert isinstance(tools_node, ToolNode)
        assert "write_todos" not in set(tools_node.tools_by_name.keys())

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


def _create_file_tool() -> CreateFileTool:
    return create_file_tool(
        AgentInternalToolResourceConfig.model_validate(
            {
                "$resourceType": "tool",
                "type": "Internal",
                "name": "Create File",
                "description": "Make a file.",
                "properties": {"toolType": "create-file"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fileName": {"type": "string"},
                        "content": {"type": "string"},
                        "filePath": {"type": "string"},
                    },
                    "required": ["fileName"],
                },
            }
        )
    )


class TestCreateFileBinding:
    def _forwarded_tool(self, tool: BaseTool, backend: Any) -> BaseTool:
        with patch(
            "uipath_langchain.agent.advanced.agent._create_deep_agent"
        ) as mock_upstream:
            mock_upstream.return_value = MagicMock(spec=CompiledStateGraph)
            create_advanced_agent(_make_mock_model(), tools=[tool], backend=backend)
        (forwarded,) = mock_upstream.call_args.kwargs["tools"]
        return forwarded

    def test_binds_the_tool_to_the_backend(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

        forwarded = self._forwarded_tool(_create_file_tool(), backend)

        assert isinstance(forwarded, CreateFileTool)
        assert forwarded.workspace is backend

    def test_keeps_an_already_bound_tool(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        bound = _create_file_tool().with_workspace(backend)

        assert self._forwarded_tool(bound, backend) is bound

    def test_leaves_the_tool_alone_without_a_backend(self) -> None:
        tool = _create_file_tool()

        assert self._forwarded_tool(tool, None) is tool
