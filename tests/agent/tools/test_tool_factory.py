"""Tests for tool_factory.py module."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models import BaseChatModel
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextSettings,
    AgentProcessToolProperties,
    AgentProcessToolResourceConfig,
    AgentSettings,
    AgentToolType,
    LowCodeAgentDefinition,
)

from uipath_langchain.agent.tools.tool_factory import create_tools_from_resources


@pytest.mark.asyncio
class TestCreateToolsFromResources:
    """Test cases for create_tools_from_resources function."""

    async def test_only_enabled_tools_returned(self):
        """Test that only enabled tools are returned from resources."""
        enabled_process_tool = AgentProcessToolResourceConfig(
            type=AgentToolType.PROCESS,
            name="EnabledProcess",
            description="Enabled process tool",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentProcessToolProperties(
                process_name="EnabledProcess",
                folder_path="/Shared/EnabledSolution",
            ),
            is_enabled=True,
        )

        disabled_context_tool = AgentContextResourceConfig(
            name="disabled_context",
            description="Disabled context tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=Mock(spec=AgentContextSettings),
            is_enabled=False,
        )

        agent = LowCodeAgentDefinition(
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            messages=[],
            settings=Mock(spec=AgentSettings),
            resources=[enabled_process_tool, disabled_context_tool],
        )

        mock_llm = AsyncMock(spec=BaseChatModel)
        tools = await create_tools_from_resources(agent, mock_llm)

        assert len(tools) == 1
        assert tools[0].name == "EnabledProcess"
