"""Tests for deeprag_tool.py module."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field
from uipath.agent.models.agent import (
    AgentContextQuerySetting,
    AgentInternalDeepRagSettings,
    AgentInternalDeepRagToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
    CitationMode,
    DeepRagCitationModeSetting,
    DeepRagFileExtension,
    DeepRagFileExtensionSetting,
)
from uipath.platform.context_grounding import (
    DeepRagResponse,
    DeepRagStatus,
    IndexStatus,
)
from uipath.platform.context_grounding.context_grounding import DeepRagContent
from uipath.platform.context_grounding.context_grounding_index import (
    ContextGroundingIndex,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.tools.internal_tools.deeprag_tool import (
    create_deeprag_tool,
)


class MockAttachment(BaseModel):
    """Mock attachment model for testing."""

    model_config = ConfigDict(populate_by_name=True)

    ID: str = Field(alias="ID")
    FullName: str = Field(alias="FullName")
    MimeType: str = Field(alias="MimeType")


class TestCreateDeepRagTool:
    """Test cases for create_deeprag_tool function."""

    @pytest.fixture
    def mock_llm(self):
        """Fixture for mock LLM."""
        return AsyncMock()

    @pytest.fixture
    def deeprag_settings_static_query(self):
        """Fixture for DeepRAG settings with static query."""
        return AgentInternalDeepRagSettings(
            context_type="attachment",
            query=AgentContextQuerySetting(
                value="What are the main points?", variant="static"
            ),
            folder_path_prefix=None,
            citation_mode=DeepRagCitationModeSetting(value=CitationMode.INLINE),
            file_extension=DeepRagFileExtensionSetting(value=DeepRagFileExtension.PDF),
        )

    @pytest.fixture
    def deeprag_settings_dynamic_query(self):
        """Fixture for DeepRAG settings with dynamic query."""
        return AgentInternalDeepRagSettings(
            context_type="attachment",
            query=AgentContextQuerySetting(
                description="Enter your query", variant="dynamic"
            ),
            folder_path_prefix=None,
            citation_mode=DeepRagCitationModeSetting(value=CitationMode.SKIP),
            file_extension=DeepRagFileExtensionSetting(value=DeepRagFileExtension.TXT),
        )

    @pytest.fixture
    def resource_config_static(self, deeprag_settings_static_query):
        """Fixture for resource configuration with static query."""
        input_schema = {
            "type": "object",
            "properties": {"attachment": {"type": "object"}},
            "required": ["attachment"],
        }
        output_schema = {"type": "object", "properties": {"text": {"type": "string"}}}

        properties = AgentInternalDeepRagToolProperties(
            tool_type=AgentInternalToolType.DEEP_RAG,
            settings=deeprag_settings_static_query,
        )

        return AgentInternalToolResourceConfig(
            name="deeprag_static",
            description="Analyze document with DeepRAG (static query)",
            input_schema=input_schema,
            output_schema=output_schema,
            properties=properties,
        )

    @pytest.fixture
    def resource_config_dynamic(self, deeprag_settings_dynamic_query):
        """Fixture for resource configuration with dynamic query."""
        input_schema = {
            "type": "object",
            "properties": {"attachment": {"type": "object"}},
            "required": ["attachment"],
        }
        output_schema = {
            "type": "object",
            "properties": {"content": {"type": "string"}},
        }

        properties = AgentInternalDeepRagToolProperties(
            tool_type=AgentInternalToolType.DEEP_RAG,
            settings=deeprag_settings_dynamic_query,
        )

        return AgentInternalToolResourceConfig(
            name="deeprag_dynamic",
            description="Analyze document with DeepRAG (dynamic query)",
            input_schema=input_schema,
            output_schema=output_schema,
            properties=properties,
        )

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_deeprag_tool_static_query_index_ready(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test DeepRAG tool with static query when index is immediately ready."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-index-123",
            last_ingestion_status="Successful",
        )

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        # Index is ready → ReadyEphemeralIndex skips interrupt() (no scratchpad in tests).
        # Only create_deeprag calls interrupt().
        deeprag_id = str(uuid.uuid4())
        mock_interrupt.side_effect = [
            DeepRagResponse(
                id=deeprag_id,
                name="test-deeprag",
                created_date="2024-01-01",
                last_deep_rag_status=DeepRagStatus.SUCCESSFUL,
                content=DeepRagContent(text="Deep RAG analysis result", citations=[]),
                failure_reason=None,
            ),
        ]

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_deeprag_tool(resource_config_static, mock_llm)

        # Verify tool creation
        assert tool.name == "deeprag_static"
        assert tool.description == "Analyze document with DeepRAG (static query)"

        # Test tool execution
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(attachment=mock_attachment)

        # Verify result
        assert result == {
            "text": "Deep RAG analysis result",
            "citations": [],
            "deepRagId": deeprag_id,
        }

        # Verify ephemeral index was created
        mock_uipath.context_grounding.create_ephemeral_index_async.assert_called_once()
        call_kwargs = (
            mock_uipath.context_grounding.create_ephemeral_index_async.call_args.kwargs
        )
        assert call_kwargs["usage"] == "DeepRAG"
        assert mock_attachment.ID in call_kwargs["attachments"]

        # Only create_deeprag calls interrupt(); index was instant-resumed
        assert mock_interrupt.call_count == 1

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_deeprag_tool_static_query_wait_for_ingestion(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test DeepRAG tool with static query when index needs to wait for ingestion."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        pending_id = str(uuid.uuid4())
        mock_index_pending = ContextGroundingIndex(
            id=pending_id,
            name="ephemeral-index-456",
            last_ingestion_status="InProgress",
        )

        mock_index_complete = {
            "id": mock_index_pending.id,
            "name": mock_index_pending.name,
            "last_ingestion_status": "Successful",
        }

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index_pending
        )

        # First interrupt returns completed index, second returns DeepRAG result
        deeprag_id = str(uuid.uuid4())
        mock_interrupt.side_effect = [
            mock_index_complete,
            DeepRagResponse(
                id=deeprag_id,
                name="test-deeprag",
                created_date="2024-01-01",
                last_deep_rag_status=DeepRagStatus.SUCCESSFUL,
                content=DeepRagContent(
                    text="Deep RAG analysis after waiting", citations=[]
                ),
                failure_reason=None,
            ),
        ]

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_deeprag_tool(resource_config_static, mock_llm)

        # Test tool execution
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(attachment=mock_attachment)

        # Verify result
        assert result == {
            "text": "Deep RAG analysis after waiting",
            "citations": [],
            "deepRagId": deeprag_id,
        }

        # Verify interrupt was called twice (WaitEphemeralIndex + CreateDeepRag)
        assert mock_interrupt.call_count == 2

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_deeprag_tool_dynamic_query(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_dynamic,
        mock_llm,
    ):
        """Test DeepRAG tool with dynamic query."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-index-789",
            last_ingestion_status="Successful",
        )

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        # Index is ready → ReadyEphemeralIndex skips interrupt(). Only create_deeprag fires.
        deeprag_id = str(uuid.uuid4())
        mock_interrupt.side_effect = [
            DeepRagResponse(
                id=deeprag_id,
                name="test-deeprag",
                created_date="2024-01-01",
                last_deep_rag_status=DeepRagStatus.SUCCESSFUL,
                content=DeepRagContent(text="Dynamic query result", citations=[]),
                failure_reason=None,
            ),
        ]

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_deeprag_tool(resource_config_dynamic, mock_llm)

        # Test tool execution with dynamic query
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.txt", MimeType="text/plain"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(
            attachment=mock_attachment, query="What is the summary?"
        )

        # Verify result
        assert result == {
            "text": "Dynamic query result",
            "citations": [],
            "deepRagId": deeprag_id,
        }

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    async def test_create_deeprag_tool_missing_attachment(
        self, mock_get_wrapper, resource_config_static, mock_llm
    ):
        """Test tool execution fails when attachment is missing."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_static, mock_llm)

        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="Argument 'attachment' is not available"):
            await tool.coroutine()

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    async def test_create_deeprag_tool_missing_query_dynamic(
        self, mock_get_wrapper, resource_config_dynamic, mock_llm
    ):
        """Test tool execution fails when query is missing (dynamic mode)."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_dynamic, mock_llm)

        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.txt", MimeType="text/plain"
        )

        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="Query is required for DeepRAG tool"):
            await tool.coroutine(attachment=mock_attachment)

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_invoke_returns_error_message_on_failed_deeprag(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test that tool returns failure_reason string when DeepRAG processing fails."""
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-index-123",
            last_ingestion_status=IndexStatus.SUCCESSFUL,
        )
        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        failed_deep_rag = DeepRagResponse(
            id=str(uuid.uuid4()),
            name="test-deeprag",
            created_date="2024-01-01",
            last_deep_rag_status=DeepRagStatus.FAILED,
            content=None,
            failure_reason="DeepRAG processing failed due to an internal error",
        )

        # Index is ready (no interrupt for index), DeepRAG fails
        mock_interrupt.side_effect = [failed_deep_rag]

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_static, mock_llm)

        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.coroutine(attachment=mock_attachment)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.DEEP_RAG_FAILED
        )
        assert "DeepRAG processing failed due to an internal error" in str(
            exc_info.value
        )
        assert mock_interrupt.call_count == 1

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_invoke_returns_error_message_on_failed_ephemeral_index(
        self,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test that tool returns failure reason when ephemeral index fails immediately."""
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-index-123",
            last_ingestion_status=IndexStatus.FAILED,
            last_ingestion_failure_reason="Ingestion failed due to unsupported file format",
        )
        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_static, mock_llm)

        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.coroutine(attachment=mock_attachment)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.EPHEMERAL_INDEX_INGESTION_FAILED
        )
        assert (
            "Attachment ingestion failed. Please check all your attachments are valid. Error: Ingestion failed due to unsupported file format"
            in str(exc_info.value)
        )

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_invoke_returns_error_message_on_failed_ephemeral_index_after_wait(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test that tool returns failure reason when ephemeral index fails after waiting."""
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        pending_id = str(uuid.uuid4())
        mock_index_pending = ContextGroundingIndex(
            id=pending_id,
            name="ephemeral-index-456",
            last_ingestion_status=IndexStatus.IN_PROGRESS,
        )
        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index_pending
        )

        mock_index_failed = {
            "id": pending_id,
            "name": mock_index_pending.name,
            "last_ingestion_status": "Failed",
            "last_ingestion_failure_reason": "Ingestion failed during processing",
        }

        # First (and only) interrupt returns the failed index; DeepRAG is never reached
        mock_interrupt.side_effect = [mock_index_failed]

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_static, mock_llm)

        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.coroutine(attachment=mock_attachment)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.EPHEMERAL_INDEX_INGESTION_FAILED
        )
        assert (
            "Attachment ingestion failed. Please check all your attachments are valid. Error: Ingestion failed during processing"
            in str(exc_info.value)
        )
        assert mock_interrupt.call_count == 1

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_failed_deeprag_with_null_failure_reason_uses_fallback_message(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test that a fallback detail message is used when DeepRAG failure_reason is None."""
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-index-123",
            last_ingestion_status=IndexStatus.SUCCESSFUL,
        )
        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        mock_interrupt.side_effect = [
            DeepRagResponse(
                id=str(uuid.uuid4()),
                name="test-deeprag",
                created_date="2024-01-01",
                last_deep_rag_status=DeepRagStatus.FAILED,
                content=None,
                failure_reason=None,
            )
        ]

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_static, mock_llm)
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.coroutine(attachment=mock_attachment)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.DEEP_RAG_FAILED
        )
        assert "Deep RAG task failed." in str(exc_info.value)

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.deeprag_tool.UiPath")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_failed_ephemeral_index_with_null_failure_reason_uses_fallback_message(
        self,
        mock_uipath_class,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test that a fallback detail message is used when ephemeral index failure_reason is None."""
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-index-123",
            last_ingestion_status=IndexStatus.FAILED,
            last_ingestion_failure_reason=None,
        )
        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_static, mock_llm)
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.coroutine(attachment=mock_attachment)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.EPHEMERAL_INDEX_INGESTION_FAILED
        )
        assert "Ephemeral index ingestion failed." in str(exc_info.value)

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    async def test_create_deeprag_tool_missing_attachment_id(
        self, mock_get_wrapper, resource_config_static, mock_llm
    ):
        """Test tool execution fails when attachment ID is missing."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_deeprag_tool(resource_config_static, mock_llm)

        class AttachmentWithoutID(BaseModel):
            FullName: str
            MimeType: str

        mock_attachment = AttachmentWithoutID(
            FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="Attachment ID is required"):
            await tool.coroutine(attachment=mock_attachment)
