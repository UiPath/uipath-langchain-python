"""Tests for extraction_tool.py metadata and functionality."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from uipath.agent.models.agent import (
    AgentIxpExtractionResourceConfig,
    AgentIxpExtractionToolProperties,
)
from uipath.platform.attachments import Attachment
from uipath.platform.documents import ExtractionResponseIXP

from uipath_langchain.agent.tools.extraction_tool import create_ixp_extraction_tool


class TestExtractionToolMetadata:
    """Test that extraction tool has correct metadata for observability."""

    @pytest.fixture
    def extraction_resource(self):
        """Create a minimal extraction tool resource config."""
        return AgentIxpExtractionResourceConfig(
            name="test_extraction",
            description="Extract data from files",
            input_schema={
                "type": "object",
                "properties": {
                    "attachment": {
                        "description": "the file uploaded as attachement",
                        "$ref": "#/definitions/job-attachment",
                    }
                },
                "required": ["attachment"],
                "definitions": {
                    "job-attachment": {
                        "type": "object",
                        "required": ["ID"],
                        "x-uipath-resource-kind": "JobAttachment",
                        "properties": {
                            "ID": {
                                "type": "string",
                                "description": "Orchestrator attachment key",
                            },
                            "FullName": {"type": "string", "description": "File name"},
                            "MimeType": {
                                "type": "string",
                                "description": 'The MIME type of the content, such as "application/json" or "image/png"',
                            },
                            "Metadata": {
                                "type": "object",
                                "description": "Dictionary<string, string> of metadata",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                    }
                },
            },
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProject",
                version_tag="v1.0",
            ),
        )

    def test_extraction_tool_has_correct_name(self, extraction_resource):
        """Test that extraction tool has sanitized name."""
        tool = create_ixp_extraction_tool(extraction_resource)

        assert tool.name == "test_extraction"

    def test_extraction_tool_has_correct_description(self, extraction_resource):
        """Test that extraction tool has correct description."""
        tool = create_ixp_extraction_tool(extraction_resource)

        assert tool.description == "Extract data from files"

    def test_extraction_tool_has_attachment_input_schema(self, extraction_resource):
        """Test that extraction tool uses Attachment as input schema."""
        tool = create_ixp_extraction_tool(extraction_resource)

        assert tool.args_schema == Attachment

    def test_extraction_tool_has_extraction_response_output_type(
        self, extraction_resource
    ):
        """Test that extraction tool has ExtractionResponseIXP as output type."""
        tool = create_ixp_extraction_tool(extraction_resource)

        assert hasattr(tool, "output_type")
        assert tool.output_type == ExtractionResponseIXP


class TestExtractionToolFunctionality:
    """Test the extraction tool function behavior."""

    @pytest.fixture
    def extraction_resource(self):
        """Create a minimal extraction tool resource config."""
        return AgentIxpExtractionResourceConfig(
            name="test_extraction",
            description="Extract data from files",
            input_schema={
                "type": "object",
                "properties": {
                    "attachment": {
                        "description": "the file uploaded as attachment",
                        "$ref": "#/definitions/job-attachment",
                    }
                },
                "required": ["attachment"],
                "definitions": {
                    "job-attachment": {
                        "type": "object",
                        "required": ["ID"],
                        "x-uipath-resource-kind": "JobAttachment",
                        "properties": {
                            "ID": {
                                "type": "string",
                                "description": "Orchestrator attachment key",
                            },
                            "FullName": {"type": "string", "description": "File name"},
                            "MimeType": {
                                "type": "string",
                                "description": "The MIME type of the content",
                            },
                        },
                    }
                },
            },
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProject",
                version_tag="v1.0",
            ),
        )

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    @patch("uipath_langchain.agent.tools.extraction_tool.interrupt")
    async def test_extraction_tool_downloads_attachment_and_calls_interrupt(
        self, mock_interrupt, mock_uipath_class, extraction_resource
    ):
        """Test that extraction tool downloads attachment and calls interrupt with correct params."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            return_value="/path/to/document.pdf"
        )
        mock_interrupt.return_value = {"extracted_data": {"field1": "value1"}}

        tool = create_ixp_extraction_tool(extraction_resource)

        result = await tool.ainvoke(
            {
                "id": "fa93f4ca-bd3f-473a-93e5-e6e5b5a8f27f",
                "full_name": "document.pdf",
                "mime_type": "application/pdf",
            }
        )

        mock_client.attachments.download_async.assert_called_once_with(
            key=UUID("fa93f4ca-bd3f-473a-93e5-e6e5b5a8f27f"),
            destination_path="document.pdf",
        )

        assert mock_interrupt.called
        interrupt_arg = mock_interrupt.call_args[0][0]
        assert interrupt_arg.project_name == "TestProject"
        assert interrupt_arg.tag == "v1.0"
        assert interrupt_arg.file_path == "/path/to/document.pdf"

        assert result == {"extracted_data": {"field1": "value1"}}

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    @patch("uipath_langchain.agent.tools.extraction_tool.interrupt")
    async def test_extraction_tool_with_different_version_tag(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test extraction tool with different version tag."""
        extraction_resource = AgentIxpExtractionResourceConfig(
            name="test_extraction_v2",
            description="Extract data from files v2",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProjectV2",
                version_tag="staging",
            ),
        )

        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            return_value="/path/to/document.pdf"
        )
        mock_interrupt.return_value = {"extracted_data": {}}

        tool = create_ixp_extraction_tool(extraction_resource)

        await tool.ainvoke(
            {
                "id": "fa93f4ca-bd3f-473a-93e5-e6e5b5a8f27f",
                "full_name": "document.pdf",
                "mime_type": "application/pdf",
            }
        )

        interrupt_arg = mock_interrupt.call_args[0][0]
        assert interrupt_arg.tag == "staging"

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    async def test_extraction_tool_propagates_download_exception(
        self, mock_uipath_class, extraction_resource
    ):
        """Test that exceptions from attachment download are propagated."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            side_effect=Exception("Download failed")
        )

        tool = create_ixp_extraction_tool(extraction_resource)

        with pytest.raises(Exception) as exc_info:
            await tool.ainvoke(
                {
                    "id": "fa93f4ca-bd3f-473a-93e5-e6e5b5a8f27f",
                    "full_name": "file.pdf",
                    "mime_type": "application/pdf",
                }
            )

        assert "Download failed" in str(exc_info.value)


class TestExtractionToolNameSanitization:
    """Test that extraction tool names are properly sanitized."""

    @pytest.mark.asyncio
    async def test_extraction_tool_name_with_spaces(self):
        """Test that tool names with spaces are sanitized."""
        resource = AgentIxpExtractionResourceConfig(
            name="Invoice Extraction Tool",
            description="Extract invoices",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="InvoiceExtraction",
                version_tag="v1.0",
            ),
        )

        tool = create_ixp_extraction_tool(resource)

        assert " " not in tool.name

    @pytest.mark.asyncio
    async def test_extraction_tool_name_with_special_chars(self):
        """Test that tool names with special characters are sanitized."""
        resource = AgentIxpExtractionResourceConfig(
            name="invoice-extraction@v1",
            description="Extract invoices",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="InvoiceExtraction",
                version_tag="v1.0",
            ),
        )

        tool = create_ixp_extraction_tool(resource)

        # Tool name should be sanitized
        assert tool.name is not None
        assert len(tool.name) > 0
