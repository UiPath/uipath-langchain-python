"""Tests for analyze_files_tool.py module."""

import json
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field
from uipath.agent.models.agent import (
    AgentInternalAnalyzeFilesToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
)

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.multimodal import FileInfo
from uipath_langchain.agent.tools.internal_tools.analyze_files_tool import (
    ANALYZE_FILES_SYSTEM_MESSAGE,
    LLM_CALL_ATTACHMENTS_METADATA_KEY,
    _config_with_llm_call_attachments,
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
        llm.model_copy = Mock(return_value=llm)
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

        properties = AgentInternalAnalyzeFilesToolProperties(
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
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.add_files_to_message"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    async def test_create_analyze_file_tool_success(
        self,
        mock_resolve_attachments,
        mock_add_files,
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

        # mock add_files_to_message to return a message with files added
        mock_message_with_files = HumanMessage(
            content=[
                {"type": "text", "text": "Summarize the document"},
                {"type": "file", "url": "https://example.com/file.pdf"},
            ]
        )
        mock_add_files.return_value = mock_message_with_files

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
            analysisTask="Summarize the document",
            attachments=[mock_attachment],
        )

        # Verify calls
        assert result == {"analysisResult": "Analyzed result"}
        mock_resolve_attachments.assert_called_once()
        mock_add_files.assert_called_once()
        mock_llm.ainvoke.assert_called_once()

        # Verify add_files_to_message was called correctly
        add_files_call_args = mock_add_files.call_args
        message_arg = add_files_call_args[0][0]
        files_arg = add_files_call_args[0][1]

        assert isinstance(message_arg, HumanMessage)
        assert message_arg.content == "Summarize the document"
        assert len(files_arg) == 1
        assert files_arg[0].url == "https://example.com/file.pdf"

        # Verify llm.ainvoke was called with correct messages
        ainvoke_call_args = mock_llm.ainvoke.call_args
        messages_arg = ainvoke_call_args[0][0]
        assert len(messages_arg) == 2
        assert messages_arg[0].content == ANALYZE_FILES_SYSTEM_MESSAGE
        assert messages_arg[1] == mock_message_with_files

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
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
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
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
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.add_files_to_message"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    async def test_create_analyze_file_tool_with_multiple_attachments(
        self,
        mock_resolve_attachments,
        mock_add_files,
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

        # mock add_files_to_message to return a message with multiple files
        mock_message_with_files = HumanMessage(
            content=[
                {"type": "text", "text": "Compare these documents"},
                {"type": "file", "url": "https://example.com/file1.pdf"},
                {"type": "file", "url": "https://example.com/file2.docx"},
            ]
        )
        mock_add_files.return_value = mock_message_with_files

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # setup llm to return analyzed result
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Multiple files analyzed")
        )

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

        assert result == {"analysisResult": "Multiple files analyzed"}
        mock_resolve_attachments.assert_called_once()

        # Verify add_files_to_message received both files
        call_args = mock_add_files.call_args
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

    async def test_resolve_attachment_with_missing_mime_type_guesses_from_filename(
        self, mock_uipath_client
    ):
        """Test resolving attachment with missing MimeType guesses from blob name."""
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
        assert result[0].mime_type == "application/pdf"

    async def test_resolve_attachment_with_none_mime_type_guesses_from_filename(
        self, mock_uipath_client
    ):
        """Test that a None MimeType attribute is handled by guessing from filename."""
        attachment_id = uuid.uuid4()

        class AttachmentWithNoneMimeType(BaseModel):
            model_config = ConfigDict(populate_by_name=True)
            ID: str = Field(alias="ID")
            FullName: str = Field(alias="FullName")
            MimeType: str | None = Field(alias="MimeType", default=None)

        mock_attachment = AttachmentWithNoneMimeType(
            ID=str(attachment_id),
            FullName="report.png",
            MimeType=None,
        )

        mock_blob_info = MockBlobInfo(
            uri="https://blob.storage.com/files/report.png",
            name="report.png",
        )

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock(
            return_value=mock_blob_info
        )

        result = await _resolve_job_attachment_arguments([mock_attachment])

        assert len(result) == 1
        assert result[0].mime_type == "image/png"

    async def test_resolve_attachment_with_empty_mime_type_guesses_from_filename(
        self, mock_uipath_client
    ):
        """Test that an empty string MimeType is handled by guessing from filename."""
        attachment_id = uuid.uuid4()

        class AttachmentWithEmptyMimeType(BaseModel):
            model_config = ConfigDict(populate_by_name=True)
            ID: str = Field(alias="ID")
            FullName: str = Field(alias="FullName")
            MimeType: str = Field(alias="MimeType")

        mock_attachment = AttachmentWithEmptyMimeType(
            ID=str(attachment_id),
            FullName="image.jpg",
            MimeType="",
        )

        mock_blob_info = MockBlobInfo(
            uri="https://blob.storage.com/files/image.jpg",
            name="image.jpg",
        )

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock(
            return_value=mock_blob_info
        )

        result = await _resolve_job_attachment_arguments([mock_attachment])

        assert len(result) == 1
        assert result[0].mime_type == "image/jpeg"

    async def test_resolve_attachment_with_no_mime_type_and_unknown_extension(
        self, mock_uipath_client
    ):
        """Test that unguessable MIME type falls back to empty string."""
        attachment_id = uuid.uuid4()

        class AttachmentWithoutMimeType(BaseModel):
            ID: str
            FullName: str

        mock_attachment = AttachmentWithoutMimeType(
            ID=str(attachment_id),
            FullName="data.xyz123",
        )

        mock_blob_info = MockBlobInfo(
            uri="https://blob.storage.com/files/data.xyz123",
            name="data.xyz123",
        )

        mock_uipath_client.attachments.get_blob_file_access_uri_async = AsyncMock(
            return_value=mock_blob_info
        )

        result = await _resolve_job_attachment_arguments([mock_attachment])

        assert len(result) == 1
        assert result[0].mime_type == ""

    async def test_resolve_attachment_with_valid_mime_type_uses_it(
        self, mock_uipath_client
    ):
        """Test that a valid MimeType from the attachment is used as-is."""
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
        assert result[0].mime_type == "application/pdf"

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


class TestCreateAnalyzeFileToolWithPiiMasking:
    """Integration tests verifying PiiMasker is wired into the tool correctly.

    Unit tests for PiiMasker itself live in test_pii_masker.py; here we assert
    that create_analyze_file_tool invokes it under the expected conditions.
    """

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Analyzed result"))
        llm.model_copy = Mock(return_value=llm)
        return llm

    @pytest.fixture
    def resource_config(self):
        input_schema = {
            "type": "object",
            "properties": {
                "analysisTask": {"type": "string"},
                "attachments": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["analysisTask", "attachments"],
        }
        output_schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        properties = AgentInternalAnalyzeFilesToolProperties(
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
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.add_files_to_message"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.analyze_files_tool.PiiMasker")
    @patch("uipath_langchain.agent.tools.internal_tools.analyze_files_tool.UiPath")
    async def test_invokes_masker_when_policy_enabled(
        self,
        mock_uipath_cls,
        mock_masker_cls,
        mock_resolve_attachments,
        mock_add_files,
        mock_get_wrapper,
        resource_config,
        mock_llm,
    ):
        mock_client = Mock()
        mock_client.automation_ops.get_deployed_policy_async = AsyncMock(
            return_value={"data": {"container": {"pii-in-flight-agents": True}}}
        )
        mock_uipath_cls.return_value = mock_client

        mock_masker_cls.is_policy_enabled = Mock(return_value=True)
        masker_instance = Mock()
        masker_instance.apply = AsyncMock(
            return_value=(
                "contact [EMAIL]",
                [
                    FileInfo(
                        url="https://redacted/doc.pdf",
                        name="pii_masked_doc.pdf",
                        mime_type="application/pdf",
                    )
                ],
            )
        )
        masker_instance.rehydrate = Mock(return_value="Sent to john@example.com")
        mock_masker_cls.return_value = masker_instance

        mock_resolve_attachments.return_value = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
            )
        ]
        mock_add_files.return_value = HumanMessage(content="contact [EMAIL]")
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Sent to [EMAIL]"))
        mock_get_wrapper.return_value = Mock()

        tool = create_analyze_file_tool(resource_config, mock_llm)
        attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="doc.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(
            analysisTask="contact john@example.com", attachments=[attachment]
        )

        mock_masker_cls.is_policy_enabled.assert_called_once_with(
            {"data": {"container": {"pii-in-flight-agents": True}}}
        )
        mock_masker_cls.assert_called_once_with(
            mock_client, {"data": {"container": {"pii-in-flight-agents": True}}}
        )
        masker_instance.apply.assert_awaited_once()
        masker_instance.rehydrate.assert_called_once_with("Sent to [EMAIL]")

        message_arg, files_arg = mock_add_files.call_args[0]
        assert message_arg.content == "contact [EMAIL]"
        assert files_arg[0].name == "pii_masked_doc.pdf"

        assert result == {"analysisResult": "Sent to john@example.com"}

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.add_files_to_message"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.analyze_files_tool.PiiMasker")
    @patch("uipath_langchain.agent.tools.internal_tools.analyze_files_tool.UiPath")
    async def test_skips_masker_when_policy_disabled(
        self,
        mock_uipath_cls,
        mock_masker_cls,
        mock_resolve_attachments,
        mock_add_files,
        mock_get_wrapper,
        resource_config,
        mock_llm,
    ):
        mock_client = Mock()
        mock_client.automation_ops.get_deployed_policy_async = AsyncMock(
            return_value={"data": {"container": {"pii-in-flight-agents": False}}}
        )
        mock_uipath_cls.return_value = mock_client
        mock_masker_cls.is_policy_enabled = Mock(return_value=False)

        mock_resolve_attachments.return_value = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
            )
        ]
        mock_add_files.return_value = HumanMessage(content="task")
        mock_get_wrapper.return_value = Mock()

        tool = create_analyze_file_tool(resource_config, mock_llm)
        attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="doc.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        await tool.coroutine(analysisTask="task", attachments=[attachment])

        # is_policy_enabled checked, but the class was never instantiated.
        mock_masker_cls.assert_not_called()

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.add_files_to_message"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.analyze_files_tool.PiiMasker")
    @patch("uipath_langchain.agent.tools.internal_tools.analyze_files_tool.UiPath")
    async def test_raises_agent_runtime_error_when_masker_apply_fails(
        self,
        mock_uipath_cls,
        mock_masker_cls,
        mock_resolve_attachments,
        mock_add_files,
        mock_get_wrapper,
        resource_config,
        mock_llm,
    ):
        mock_client = Mock()
        mock_client.automation_ops.get_deployed_policy_async = AsyncMock(
            return_value={"data": {"container": {"pii-in-flight-agents": True}}}
        )
        mock_uipath_cls.return_value = mock_client

        mock_masker_cls.is_policy_enabled = Mock(return_value=True)
        underlying = RuntimeError("proxy unavailable")
        masker_instance = Mock()
        masker_instance.apply = AsyncMock(side_effect=underlying)
        mock_masker_cls.return_value = masker_instance

        mock_resolve_attachments.return_value = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
            )
        ]
        mock_get_wrapper.return_value = Mock()

        tool = create_analyze_file_tool(resource_config, mock_llm)
        attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="doc.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.coroutine(analysisTask="task", attachments=[attachment])

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.UNEXPECTED_ERROR
        )
        assert exc_info.value.__cause__ is underlying
        mock_llm.ainvoke.assert_not_called()
        mock_add_files.assert_not_called()


class TestConfigWithLlmCallAttachments:
    """The attachments payload travels to the llmCall span via langchain config metadata."""

    def test_returns_config_unchanged_when_no_files(self) -> None:
        config: RunnableConfig = {"tags": ["existing"]}
        assert _config_with_llm_call_attachments(config, []) is config

    def test_returns_none_when_input_is_none_and_no_files(self) -> None:
        assert _config_with_llm_call_attachments(None, []) is None

    def test_injects_payload_into_metadata(self) -> None:
        att_id = str(uuid.uuid4())
        files = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
                attachment_id=att_id,
            )
        ]

        new_config = _config_with_llm_call_attachments(None, files)
        assert new_config is not None
        payload = new_config["metadata"][LLM_CALL_ATTACHMENTS_METADATA_KEY]
        attachments = json.loads(payload)
        assert len(attachments) == 1
        assert attachments[0]["id"] == att_id
        assert attachments[0]["fileName"] == "doc.pdf"

    def test_preserves_existing_metadata_keys(self) -> None:
        files = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
                attachment_id=str(uuid.uuid4()),
            )
        ]
        config: RunnableConfig = {"metadata": {"unrelated": "value"}, "tags": ["t"]}

        new_config = _config_with_llm_call_attachments(config, files)
        assert new_config is not None
        assert new_config["metadata"]["unrelated"] == "value"
        assert LLM_CALL_ATTACHMENTS_METADATA_KEY in new_config["metadata"]
        assert new_config["tags"] == ["t"]

    def test_uses_masked_attachment_when_present(self) -> None:
        masked_id = str(uuid.uuid4())
        files = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
                attachment_id=str(uuid.uuid4()),
                masked_attachment_url="https://redacted/doc.pdf",
                masked_attachment_id=masked_id,
            )
        ]

        new_config = _config_with_llm_call_attachments({}, files)
        assert new_config is not None
        payload = new_config["metadata"][LLM_CALL_ATTACHMENTS_METADATA_KEY]
        attachments = json.loads(payload)
        assert attachments[0]["id"] == masked_id
        assert attachments[0]["fileName"] == "pii_masked_doc.pdf"


class TestAnalyzeFileToolPassesAttachmentsViaConfig:
    """End-to-end: tool injects attachments metadata into the config given to ainvoke."""

    @pytest.fixture
    def resource_config(self):
        input_schema = {
            "type": "object",
            "properties": {
                "analysisTask": {"type": "string"},
                "attachments": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["analysisTask", "attachments"],
        }
        output_schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        properties = AgentInternalAnalyzeFilesToolProperties(
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
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool.add_files_to_message"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.analyze_files_tool._resolve_job_attachment_arguments"
    )
    async def test_ainvoke_called_with_attachments_metadata(
        self,
        mock_resolve_attachments,
        mock_add_files,
        mock_get_wrapper,
        resource_config,
    ) -> None:
        att_id = str(uuid.uuid4())
        mock_resolve_attachments.return_value = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
                attachment_id=att_id,
            )
        ]
        mock_add_files.return_value = HumanMessage(content="task")
        mock_get_wrapper.return_value = Mock()

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="result"))
        llm.model_copy = Mock(return_value=llm)

        tool = create_analyze_file_tool(resource_config, llm)
        attachment = MockAttachment(
            ID=att_id, FullName="doc.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        await tool.coroutine(analysisTask="task", attachments=[attachment])

        config_arg = llm.ainvoke.call_args.kwargs["config"]
        assert config_arg is not None
        payload = config_arg["metadata"][LLM_CALL_ATTACHMENTS_METADATA_KEY]
        attachments = json.loads(payload)
        assert attachments[0]["id"] == att_id
        assert attachments[0]["fileName"] == "doc.pdf"
