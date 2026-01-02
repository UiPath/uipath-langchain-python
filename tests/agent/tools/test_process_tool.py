"""Tests for process_tool.py metadata."""

import pytest
from uipath.agent.models.agent import (
    AgentProcessToolProperties,
    AgentProcessToolResourceConfig,
    AgentToolType,
)

from uipath_langchain.agent.tools.process_tool import create_process_tool


class TestProcessToolMetadata:
    """Test that process tool has correct metadata for observability."""

    @pytest.fixture
    def process_resource(self):
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

    def test_process_tool_has_metadata(self, process_resource):
        """Test that process tool has metadata dict."""
        tool = create_process_tool(process_resource)

        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    def test_process_tool_metadata_has_tool_type(self, process_resource):
        """Test that metadata contains tool_type for span detection."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "process"

    def test_process_tool_metadata_has_display_name(self, process_resource):
        """Test that metadata contains display_name from process_name."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "MyProcess"

    def test_process_tool_metadata_has_folder_path(self, process_resource):
        """Test that metadata contains folder_path for span attributes."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"
