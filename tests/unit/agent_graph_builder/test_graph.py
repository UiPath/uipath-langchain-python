"""Tests for graph building functionality."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentMessage,
    AgentSettings,
    LowCodeAgentDefinition,
)

from uipath_lowcode.agent_graph_builder import build_agent_graph


def create_test_agent_definition(**overrides: Any) -> LowCodeAgentDefinition:
    """Create a test agent definition."""
    defaults = {
        "id": "test-agent",
        "name": "Test Agent",
        "messages": [
            AgentMessage(role="system", content="Test system message"),
            AgentMessage(role="user", content="Test user message"),
        ],
        "settings": AgentSettings(
            engine="azure_openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1024,
        ),
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object"},
        "resources": [],
    }
    defaults.update(overrides)
    return LowCodeAgentDefinition(**defaults)


@pytest.mark.asyncio
class TestBuildAgentGraph:
    """Test building the agent graph."""

    async def test_builds_graph_with_minimal_config(self):
        """Test that graph is built with minimal configuration."""
        agent_def = create_test_agent_definition()

        with (
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            graph = await build_agent_graph(agent_def)

            assert graph is not None
            mock_create.assert_called_once()

    async def test_passes_llm_settings(self):
        """Test that LLM settings are passed correctly."""
        agent_def = create_test_agent_definition(
            settings=AgentSettings(
                engine="azure_openai",
                model="test-model",
                temperature=0.5,
                max_tokens=2048,
            )
        )

        with (
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_lowcode.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_agent"
            ) as mock_create,
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_llm.assert_called_once_with(
                model="test-model",
                temperature=0.5,
                max_tokens=2048,
            )

    async def test_handles_input_data_dict(self):
        """Test building graph with dict input data."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(role="system", content="Process {{task}}"),
                AgentMessage(role="user", content="Start"),
            ],
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        )

        with (
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def, input_data={"task": "analysis"})

            mock_create.assert_called_once()
            messages = mock_create.call_args.kwargs["messages"]
            assert "analysis" in messages[0].content

    async def test_handles_input_data_json_string(self):
        """Test building graph with JSON string input data."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(role="system", content="Count: {{count}}"),
                AgentMessage(role="user", content="Start"),
            ],
            input_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        )

        with (
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def, input_data='{"count": 42}')

            mock_create.assert_called_once()
            messages = mock_create.call_args.kwargs["messages"]
            assert "42" in messages[0].content

    async def test_creates_tools_from_resources(self):
        """Test that tools are created from agent resources."""
        agent_def = create_test_agent_definition(
            resources=[
                {
                    "type": "tool",
                    "name": "test_tool",
                    "description": "A test tool",
                    "$resourceType": "custom",
                }
            ]
        )

        mock_tool = MagicMock()
        with (
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ) as mock_create_tools,
            patch(
                "uipath_lowcode.agent_graph_builder.graph.create_agent"
            ) as mock_create_agent,
        ):
            mock_create_agent.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create_tools.assert_called_once()
            mock_create_agent.assert_called_once()
            call_kwargs = mock_create_agent.call_args.kwargs
            assert call_kwargs["tools"] == [mock_tool]
