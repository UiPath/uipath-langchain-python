"""Tests for build_agent_graph taking the deep agent path (_build_deep_agent)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.structured_output import ToolStrategy
from uipath.agent.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentSettings,
    LowCodeAgentDefinition,
)

from uipath_agents.agent_graph_builder import build_agent_graph


def _make_deep_agent_definition(**overrides: Any) -> LowCodeAgentDefinition:
    """Create a test agent definition with deepAgent=True."""
    defaults: dict[str, Any] = {
        "id": "deep-test",
        "name": "Deep Agent",
        "messages": [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="You are a deep agent."),
            AgentMessage(role=AgentMessageRole.USER, content="Do {{task}}"),
        ],
        "settings": AgentSettings(
            engine="basic-v2",
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            temperature=0,
            max_tokens=32000,
        ),
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        },
        "resources": [],
    }
    defaults.update(overrides)
    agent_def = LowCodeAgentDefinition(**defaults)
    # Pydantic rejects MagicMock during validation, so mock post-construction
    agent_def.settings = MagicMock(deepAgent={"enabled": True})
    return agent_def


@pytest.mark.asyncio
class TestBuildDeepAgentGraph:
    @pytest.fixture(autouse=True)
    def mock_mcp_tools(self) -> Any:
        with patch(
            "uipath_agents.agent_graph_builder.graph.create_mcp_tools_and_clients",
            new_callable=AsyncMock,
            return_value=([], []),
        ):
            yield

    async def test_calls_create_deep_agent_when_enabled(self) -> None:
        agent_def = _make_deep_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_deep_agent_graph"
            ) as mock_create_deep_graph,
        ):
            mock_llm.return_value = MagicMock()
            mock_create_deep_graph.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create_deep_graph.assert_called_once()

    async def test_does_not_call_create_agent_when_deep(self) -> None:
        agent_def = _make_deep_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_deep_agent_graph"
            ) as mock_create_deep_graph,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create_shallow,
        ):
            mock_llm.return_value = MagicMock()
            mock_create_deep_graph.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create_shallow.assert_not_called()

    async def test_passes_response_format(self) -> None:
        agent_def = _make_deep_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_deep_agent_graph"
            ) as mock_create_deep_graph,
        ):
            mock_llm.return_value = MagicMock()
            mock_create_deep_graph.return_value = MagicMock()

            await build_agent_graph(agent_def)

            call_kwargs = mock_create_deep_graph.call_args.kwargs
            assert isinstance(call_kwargs["response_format"], ToolStrategy)

    async def test_passes_tools_through(self) -> None:
        agent_def = _make_deep_agent_definition()
        mock_tool = MagicMock()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_deep_agent_graph"
            ) as mock_create_deep_graph,
        ):
            mock_llm.return_value = MagicMock()
            mock_create_deep_graph.return_value = MagicMock()

            await build_agent_graph(agent_def)

            call_kwargs = mock_create_deep_graph.call_args.kwargs
            assert call_kwargs["tools"] == [mock_tool]

    async def test_passes_state_backend(self) -> None:
        # backend=None tells deepagents to use the default StateBackend
        agent_def = _make_deep_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_deep_agent_graph"
            ) as mock_create_deep_graph,
        ):
            mock_llm.return_value = MagicMock()
            mock_create_deep_graph.return_value = MagicMock()

            await build_agent_graph(agent_def)

            call_kwargs = mock_create_deep_graph.call_args.kwargs
            assert call_kwargs["backend"] is None
