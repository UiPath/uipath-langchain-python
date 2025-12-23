"""Tests for analyze_files_tool.py module."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field
from uipath.agent.models.agent import (
    AgentInternalToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
)

from uipath_langchain.agent.react.llm_with_files import FileInfo
from uipath_langchain.agent.tools.internal_tools.analyze_files_tool import (
    ANALYZE_FILES_SYSTEM_MESSAGE,
    _resolve_job_attachment_arguments,
    create_analyze_file_tool,
)


class MockAttachment(BaseModel):
    """Mock attachment model for testing."""

    model_config = ConfigDict(populate_by_name=True)

    ID: str = Field(alias="ID")
    FullName: str = Field(alias="FullName")
    MimeType: str = Field(alias="MimeType")


class MockBlobInfo(BaseModel):
    """Mock blob info model for testing."""

    uri: str
    name: str


class TestCreateAnalyzeFileTool:
    """Test cases for create_analyze_file_tool function."""

    @pytest.fixture
    def mock_llm(self):
        """Fixture for mock LLM."""
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Analyzed result"))
        return llm

    @pytest.fixture
    def resource_config(self):
        """Fixture for resource configuration."""
        input_schema = {
            "type": "object",
            "properties": {
                "analysisTask": {"type": "string"},
                "attachments": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["analysisTask", "attachments"],
        }
        output_schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        properties = AgentInternalToolProperties(
            tool_type=AgentInternalToolType.ANALYZE_FILES
        )

        return AgentInternalToolResourceConfig(
            name="analyze_files",
            description="Analyze files with AI",
            input_schema=input_schema,
            output_schema=output_schema,
            properties=properties,
        )

    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.llm_call_with_files"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    async def test_create_analyze_file_tool_success(
        self,
        mock_resolve_attachments,
        mock_llm_call,
        mock_get_wrapper,
        resource_config,
        mock_llm,
    ):
        """Test successful creation and execution of analyze file tool."""
        # Setup mocks
        mock_resolve_attachments.return_value = [
            FileInfo(
                url="https://example.com/file.pdf",
                name="test.pdf",
                mime_type="application/pdf",
            )
        ]
        mock_llm_call.return_value = "Analysis complete"
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_analyze_file_tool(resource_config, mock_llm)

        # Verify tool creation
        assert tool.name == "analyze_files"
        assert tool.description == "Analyze files with AI"
        assert hasattr(tool, "coroutine")

        # Test tool execution
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(
            analysisTask="Summarize the document", attachments=[mock_attachment]
        )

        # Verify calls
        assert result == "Analysis complete"
        mock_resolve_attachments.assert_called_once()
        mock_llm_call.assert_called_once()

        # Verify LLM call arguments
        call_args = mock_llm_call.call_args
        messages, files, llm = call_args[0]
        assert len(messages) == 2
        assert messages[0].content == ANALYZE_FILES_SYSTEM_MESSAGE
        assert messages[1].content == "Summarize the document"
        assert len(files) == 1
        assert files[0].url == "https://example.com/file.pdf"

    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.get_job_attachment_wrapper"
    )
    async def test_create_analyze_file_tool_missing_analysis_task(
        self, mock_get_wrapper, resource_config, mock_llm
    ):
        """Test tool execution fails when analysisTask is missing."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_analyze_file_tool(resource_config, mock_llm)

        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(
            ValueError, match="Argument 'analysisTask' is not available"
        ):
            await tool.coroutine(attachments=[mock_attachment])

    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.get_job_attachment_wrapper"
    )
    async def test_create_analyze_file_tool_missing_attachments(
        self, mock_get_wrapper, resource_config, mock_llm
    ):
        """Test tool execution fails when attachments are missing."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_analyze_file_tool(resource_config, mock_llm)

        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="Argument 'attachments' is not available"):
            await tool.coroutine(analysisTask="Summarize the document")

    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.llm_call_with_files"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    async def test_create_analyze_file_tool_with_multiple_attachments(
        self,
        mock_resolve_attachments,
        mock_llm_call,
        mock_get_wrapper,
        resource_config,
        mock_llm,
    ):
        """Test tool execution with multiple attachments."""
        mock_resolve_attachments.return_value = [
            FileInfo(
                url="https://example.com/file1.pdf",
                name="doc1.pdf",
                mime_type="application/pdf",
            ),
            FileInfo(
                url="https://example.com/file2.docx",
                name="doc2.docx",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ]
        mock_llm_call.return_value = "Multiple files analyzed"
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_analyze_file_tool(resource_config, mock_llm)

        mock_attachments = [
            MockAttachment(
                ID=str(uuid.uuid4()), FullName="doc1.pdf", MimeType="application/pdf"
            ),
            MockAttachment(
                ID=str(uuid.uuid4()),
                FullName="doc2.docx",
                MimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ]

        assert tool.coroutine is not None
        result = await tool.coroutine(
            analysisTask="Compare these documents", attachments=mock_attachments
        )

        assert result == "Multiple files analyzed"
        mock_resolve_attachments.assert_called_once()

        # Verify LLM received both files
        call_args = mock_llm_call.call_args
        files = call_args[0][1]
        assert len(files) == 2


class TestResolveJobAttachmentArguments:
    """Test cases for _resolve_job_attachment_arguments function."""

    @pytest.fixture
    def mock_uipath_client(self):
        """Fixture for mock UiPath client."""
        with patch(
            "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.UiPath"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            yield mock_client

    async def test_resolve_single_attachment(self, mock_uipath_client):
        """Test resolving a single attachment."""
        attachment_id = uuid.uuid4()
        mock_attachment = MockAttachment(
            ID=str(attachment_id),
            FullName="document.pdf",
            MimeType="application/pdf",
        )

        mock_blob_info = MockBlobInfo(
            uri="https://blob.storage.com/files/document.pdf",
            name="document.pdf",
        )

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock(
            return_value=mock_blob_info
        )

        result = await _resolve_job_attachment_arguments([mock_attachment])

        assert len(result) == 1
        assert result[0].url == "https://blob.storage.com/files/document.pdf"
        assert result[0].name == "document.pdf"
        assert result[0].mime_type == "application/pdf"

        mock_uipath_client.attachments.get_blob_file_access_uri_async.assert_called_once_with(
            key=attachment_id
        )

    async def test_resolve_multiple_attachments(self, mock_uipath_client):
        """Test resolving multiple attachments."""
        attachment_id_1 = uuid.uuid4()
        attachment_id_2 = uuid.uuid4()

        mock_attachments = [
            MockAttachment(
                ID=str(attachment_id_1),
                FullName="doc1.pdf",
                MimeType="application/pdf",
            ),
            MockAttachment(
                ID=str(attachment_id_2),
                FullName="image.png",
                MimeType="image/png",
            ),
        ]

        mock_blob_infos = [
            MockBlobInfo(
                uri="https://blob.storage.com/files/doc1.pdf", name="doc1.pdf"
            ),
            MockBlobInfo(
                uri="https://blob.storage.com/files/image.png", name="image.png"
            ),
        ]

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock(
            side_effect=mock_blob_infos
        )

        result = await _resolve_job_attachment_arguments(mock_attachments)

        assert len(result) == 2
        assert result[0].url == "https://blob.storage.com/files/doc1.pdf"
        assert result[0].name == "doc1.pdf"
        assert result[0].mime_type == "application/pdf"
        assert result[1].url == "https://blob.storage.com/files/image.png"
        assert result[1].name == "image.png"
        assert result[1].mime_type == "image/png"

        assert (
            mock_uipath_client.attachments.get_blob_file_access_uri_async.call_count
            == 2
        )

    async def test_resolve_attachment_without_id_skips(self, mock_uipath_client):
        """Test that attachments without ID are skipped."""

        class AttachmentWithoutID(BaseModel):
            FullName: str
            MimeType: str

        mock_attachments = [
            AttachmentWithoutID(FullName="doc.pdf", MimeType="application/pdf"),
        ]

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock()

        result = await _resolve_job_attachment_arguments(mock_attachments)

        assert len(result) == 0
        mock_uipath_client.attachments.get_blob_file_access_uri_async.assert_not_called()

    async def test_resolve_empty_attachments_list(self, mock_uipath_client):
        """Test resolving an empty list of attachments."""
        result = await _resolve_job_attachment_arguments([])

        assert len(result) == 0

    async def test_resolve_attachment_with_missing_mime_type(self, mock_uipath_client):
        """Test resolving attachment with missing MimeType defaults to empty string."""
        attachment_id = uuid.uuid4()

        class AttachmentWithoutMimeType(BaseModel):
            ID: str
            FullName: str

        mock_attachment = AttachmentWithoutMimeType(
            ID=str(attachment_id),
            FullName="document.pdf",
        )

        mock_blob_info = MockBlobInfo(
            uri="https://blob.storage.com/files/document.pdf",
            name="document.pdf",
        )

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock(
            return_value=mock_blob_info
        )

        result = await _resolve_job_attachment_arguments([mock_attachment])

        assert len(result) == 1
        assert result[0].mime_type == ""

    async def test_resolve_attachment_with_invalid_uuid_raises(
        self, mock_uipath_client
    ):
        """Test that invalid UUID in ID field raises ValueError."""

        class AttachmentWithInvalidID(BaseModel):
            ID: str
            FullName: str
            MimeType: str

        mock_attachment = AttachmentWithInvalidID(
            ID="not-a-valid-uuid",
            FullName="document.pdf",
            MimeType="application/pdf",
        )

        with pytest.raises(ValueError):
            await _resolve_job_attachment_arguments([mock_attachment])

    async def test_resolve_attachments_mixed_valid_and_invalid(
        self, mock_uipath_client
    ):
        """Test resolving mix of valid attachments and attachments without IDs."""
        attachment_id = uuid.uuid4()

        class AttachmentWithoutID(BaseModel):
            FullName: str
            MimeType: str

        mock_attachments = [
            MockAttachment(
                ID=str(attachment_id),
                FullName="doc1.pdf",
                MimeType="application/pdf",
            ),
            AttachmentWithoutID(FullName="doc2.pdf", MimeType="application/pdf"),
        ]

        mock_blob_info = MockBlobInfo(
            uri="https://blob.storage.com/files/doc1.pdf",
            name="doc1.pdf",
        )

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock(
            return_value=mock_blob_info
        )

        result = await _resolve_job_attachment_arguments(mock_attachments)

        # Only the valid attachment should be resolved
        assert len(result) == 1
        assert result[0].url == "https://blob.storage.com/files/doc1.pdf"
        mock_uipath_client.attachments.get_blob_file_access_uri_async.assert_called_once()
