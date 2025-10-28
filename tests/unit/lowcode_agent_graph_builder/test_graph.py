"""Tests for graph building functionality."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uipath_lowcode.lowcode_agent_graph_builder.constants import (
    AGENT_CONFIG_FILENAME,
)
from uipath_lowcode.lowcode_agent_graph_builder.graph import (
    build_lowcode_agent_graph,
)


def create_valid_agent_config(**overrides: Any) -> dict[str, Any]:
    """Create a valid minimal agent configuration for testing."""
    config: dict[str, Any] = {
        "id": "test-agent",
        "name": "Test Agent",
        "messages": [{"role": "system", "content": "test"}],
        "settings": {
            "engine": "azure_openai",
            "model": "gpt-4",
            "maxTokens": 4096,
            "temperature": 0.7,
        },
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object"},
        "resources": [],
    }
    for key, value in overrides.items():
        if key == "settings" and isinstance(value, dict):
            config["settings"].update(value)
        else:
            config[key] = value
    return config


@pytest.mark.asyncio
class TestBuildLowcodeAgentGraph:
    """Test building the lowcode agent graph."""

    async def test_loads_configuration_from_cwd(self, tmp_path: Path, monkeypatch):
        """Test that configuration is loaded from current working directory."""
        monkeypatch.chdir(tmp_path)

        config = create_valid_agent_config(
            messages=[{"role": "system", "content": "Test system message"}],
            settings={"model": "test-model", "temperature": 0.5, "maxTokens": 1024},
        )
        config_file = tmp_path / AGENT_CONFIG_FILENAME
        config_file.write_text(json.dumps(config))

        with (
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_lowcode_agent"
            ) as mock_create,
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_llm"
            ) as mock_llm,
        ):
            mock_create.return_value = MagicMock()
            mock_llm.return_value = MagicMock()

            await build_lowcode_agent_graph()

            mock_llm.assert_called_once()
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["model"] == "test-model"
            assert call_kwargs["temperature"] == 0.5

    async def test_handles_input_data_dict(self, tmp_path: Path, monkeypatch):
        """Test building graph with dict input data."""
        monkeypatch.chdir(tmp_path)

        config = create_valid_agent_config(
            messages=[{"role": "system", "content": "Process {{task}}"}],
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        )
        config_file = tmp_path / AGENT_CONFIG_FILENAME
        config_file.write_text(json.dumps(config))

        with (
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_lowcode_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            await build_lowcode_agent_graph(input_data={"task": "analysis"})

            mock_create.assert_called_once()
            messages = mock_create.call_args.kwargs["messages"]
            assert "analysis" in messages[0].content

    async def test_handles_input_data_json_string(self, tmp_path: Path, monkeypatch):
        """Test building graph with JSON string input data."""
        monkeypatch.chdir(tmp_path)

        config = create_valid_agent_config(
            messages=[{"role": "system", "content": "Count: {{count}}"}],
            input_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        )
        config_file = tmp_path / AGENT_CONFIG_FILENAME
        config_file.write_text(json.dumps(config))

        with (
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_lowcode_agent"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock()

            await build_lowcode_agent_graph(input_data='{"count": 42}')

            mock_create.assert_called_once()
            messages = mock_create.call_args.kwargs["messages"]
            assert "42" in messages[0].content

    async def test_creates_tools_from_resources(self, tmp_path: Path, monkeypatch):
        """Test that tools are created from agent resources."""
        monkeypatch.chdir(tmp_path)

        config = create_valid_agent_config(
            resources=[{"type": "tool", "name": "test_tool"}]
        )
        config_file = tmp_path / AGENT_CONFIG_FILENAME
        config_file.write_text(json.dumps(config))

        mock_tool = MagicMock()
        with (
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ) as mock_create_tools,
            patch(
                "uipath_lowcode.lowcode_agent_graph_builder.graph.create_lowcode_agent"
            ) as mock_create_agent,
        ):
            mock_create_agent.return_value = MagicMock()

            await build_lowcode_agent_graph()

            mock_create_tools.assert_called_once()
            mock_create_agent.assert_called_once()
            call_kwargs = mock_create_agent.call_args.kwargs
            assert call_kwargs["tools"] == [mock_tool]
