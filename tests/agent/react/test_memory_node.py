"""Tests for memory recall node and memory integration in create_agent."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables.graph import Edge
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.memory_node import (
    _build_search_fields,
    create_memory_recall_node,
)
from uipath_langchain.agent.react.types import (
    AgentGraphNode,
    MemoryConfig,
)


class _TopicInput(BaseModel):
    """Reusable test input schema."""

    topic: str = Field(default="")


class _TopicLevelInput(BaseModel):
    """Reusable test input schema with two fields."""

    topic: str = Field(default="")
    level: str = Field(default="")


def _make_state(**user_fields: Any) -> dict[str, Any]:
    """Create a state dict with standard internal fields + user fields."""
    return {"messages": [], "inner_state": {}, **user_fields}


class TestBuildSearchFields:
    def test_basic_fields(self) -> None:
        fields = _build_search_fields({"topic": "python", "level": "advanced"})
        assert len(fields) == 2
        key_paths = [f.key_path for f in fields]
        assert ["agent-input", "topic"] in key_paths
        assert ["agent-input", "level"] in key_paths

    def test_filters_none_and_uipath_prefix(self) -> None:
        fields = _build_search_fields(
            {"topic": "py", "uipath__settings": {}, "empty": None}
        )
        assert len(fields) == 1
        assert fields[0].key_path == ["agent-input", "topic"]

    def test_empty_input(self) -> None:
        assert _build_search_fields({}) == []


class TestMemoryRecallNode:
    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_returns_injection_on_success(
        self, mock_uipath_cls: MagicMock
    ) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_response = MagicMock()
        mock_response.results = [MagicMock()]
        mock_response.system_prompt_injection = "\n\nBased on past runs..."
        mock_sdk.memory.search_async = AsyncMock(return_value=mock_response)

        config = MemoryConfig(memory_space_id="space-123", field_weights={"topic": 1.0})
        node = create_memory_recall_node(config, input_schema=_TopicInput)

        result = await node(_make_state(topic="python"))
        assert result["inner_state"]["memory_injection"] == "\n\nBased on past runs..."

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_returns_empty_on_failure(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.search_async = AsyncMock(side_effect=Exception("fail"))

        config = MemoryConfig(memory_space_id="space-123", field_weights={"topic": 1.0})
        node = create_memory_recall_node(config, input_schema=_TopicInput)

        result = await node(_make_state(topic="python"))
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_user_inputs(self) -> None:
        config = MemoryConfig(memory_space_id="space-123", field_weights={"topic": 1.0})
        node = create_memory_recall_node(config, input_schema=_TopicInput)

        result = await node(_make_state())
        assert result == {}


class TestCreateAgentWithMemory:
    """Test that create_agent adds MEMORY_RECALL node when memory config is provided."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        return MagicMock(spec=BaseChatModel)

    @pytest.fixture
    def messages(self) -> list[SystemMessage]:
        return [SystemMessage(content="You are a helpful assistant.")]

    @pytest.fixture
    def memory_config(self) -> MemoryConfig:
        return MemoryConfig(
            memory_space_id="test-space-id", field_weights={"topic": 1.0}
        )

    def test_graph_has_memory_recall_node(
        self,
        mock_model: MagicMock,
        messages: list[SystemMessage],
        memory_config: MemoryConfig,
    ) -> None:
        result: StateGraph[Any] = create_agent(
            mock_model, [], messages, memory=memory_config
        )
        graph = result.compile().get_graph()
        assert AgentGraphNode.MEMORY_RECALL in graph.nodes

    def test_graph_edges_with_memory(
        self,
        mock_model: MagicMock,
        messages: list[SystemMessage],
        memory_config: MemoryConfig,
    ) -> None:
        result: StateGraph[Any] = create_agent(
            mock_model, [], messages, memory=memory_config
        )
        graph = result.compile().get_graph()
        assert Edge("__start__", AgentGraphNode.MEMORY_RECALL) in graph.edges
        assert Edge(AgentGraphNode.MEMORY_RECALL, AgentGraphNode.INIT) in graph.edges
        assert Edge("__start__", AgentGraphNode.INIT) not in graph.edges

    def test_graph_without_memory_has_no_recall_node(
        self, mock_model: MagicMock, messages: list[SystemMessage]
    ) -> None:
        result: StateGraph[Any] = create_agent(mock_model, [], messages)
        graph = result.compile().get_graph()
        assert AgentGraphNode.MEMORY_RECALL not in graph.nodes
        assert Edge("__start__", AgentGraphNode.INIT) in graph.edges
