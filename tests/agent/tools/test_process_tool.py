"""Tests for process_tool.py — metadata and tool creation."""

import pytest
from uipath.agent.models.agent import (
    AgentProcessToolProperties,
    AgentProcessToolResourceConfig,
    AgentToolType,
)

from uipath_langchain.agent.tools.process_tool import create_process_tool


@pytest.fixture
def process_resource() -> AgentProcessToolResourceConfig:
    """Create a minimal process tool resource config."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.PROCESS,
        name="test_process",
        description="Test process description",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        properties=AgentProcessToolProperties(
            process_name="MyProcess",
            folder_path="/Shared/MyFolder",
        ),
    )


@pytest.fixture
def process_resource_with_input() -> AgentProcessToolResourceConfig:
    """Create a process tool resource with an input schema."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.PROCESS,
        name="process_with_input",
        description="Process with input schema",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
            },
        },
        properties=AgentProcessToolProperties(
            process_name="InputProcess",
            folder_path="/Shared/InputFolder",
        ),
    )


class TestProcessToolCreation:
    """Test process tool creation, metadata, and structural properties."""

    def test_tool_properties_and_metadata(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)

        assert tool.name == "test_process"
        assert tool.description == "Test process description"
        assert tool.coroutine is not None
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "process"
        assert tool.metadata["display_name"] == "MyProcess"
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"

    def test_tool_name_sanitized_for_special_chars(self) -> None:
        resource = AgentProcessToolResourceConfig(
            type=AgentToolType.PROCESS,
            name="my process (v2)!",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentProcessToolProperties(
                process_name="SanitizeMe",
                folder_path="/Shared",
            ),
        )
        tool = create_process_tool(resource)
        assert " " not in tool.name
        assert "(" not in tool.name
        assert "!" not in tool.name

    def test_tool_with_input_schema(
        self, process_resource_with_input: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource_with_input)
        assert tool.args_schema is not None
        assert tool.metadata is not None
        assert "args_schema" in tool.metadata

    def test_none_folder_path_in_metadata(self) -> None:
        resource = AgentProcessToolResourceConfig(
            type=AgentToolType.PROCESS,
            name="no_folder",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentProcessToolProperties(
                process_name="NoFolderProcess",
                folder_path=None,
            ),
        )
        tool = create_process_tool(resource)
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] is None
