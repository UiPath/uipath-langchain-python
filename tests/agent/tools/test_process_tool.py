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


class TestProcessToolMetadata:
    """Test that process tool has correct metadata for observability."""

    def test_process_tool_has_metadata(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)

        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    def test_process_tool_metadata_has_tool_type(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "process"

    def test_process_tool_metadata_has_display_name(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "MyProcess"

    def test_process_tool_metadata_has_folder_path(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"


class TestProcessToolCreation:
    """Test process tool structural properties."""

    def test_tool_name_matches_resource_name(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)
        assert tool.name == "test_process"

    def test_tool_description_matches_resource(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)
        assert tool.description == "Test process description"

    def test_tool_is_async(
        self, process_resource: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource)
        assert tool.coroutine is not None

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

    def test_tool_with_input_schema_has_args_schema(
        self, process_resource_with_input: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource_with_input)
        assert tool.args_schema is not None

    def test_tool_metadata_includes_args_schema(
        self, process_resource_with_input: AgentProcessToolResourceConfig
    ) -> None:
        tool = create_process_tool(process_resource_with_input)
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
