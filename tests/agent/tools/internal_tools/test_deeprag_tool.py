"""Tests for deeprag_tool.py module."""

import os
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
from uipath.platform.common import CreateDeepRag

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
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_deeprag_tool_static_query(
        self,
        mock_interrupt,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Static-query path emits a single CreateDeepRag interrupt with the configured prompt."""
        mock_interrupt.side_effect = [{"text": "Deep RAG analysis result"}]
        mock_get_wrapper.return_value = Mock()

        tool = create_deeprag_tool(resource_config_static, mock_llm)

        assert tool.name == "deeprag_static"
        assert tool.description == "Analyze document with DeepRAG (static query)"

        attachment_id = str(uuid.uuid4())
        mock_attachment = MockAttachment(
            ID=attachment_id, FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(attachment=mock_attachment)

        assert result == {"text": "Deep RAG analysis result"}

        assert mock_interrupt.call_count == 1
        create_payload = mock_interrupt.call_args.args[0]
        assert isinstance(create_payload, CreateDeepRag)
        assert create_payload.attachments == [attachment_id]
        assert create_payload.prompt == "What are the main points?"
        assert create_payload.citation_mode == CitationMode.INLINE

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_deeprag_tool_dynamic_query(
        self,
        mock_interrupt,
        mock_get_wrapper,
        resource_config_dynamic,
        mock_llm,
    ):
        """Dynamic-query path threads the caller's query into the CreateDeepRag prompt."""
        mock_interrupt.side_effect = [{"content": "Dynamic query result"}]
        mock_get_wrapper.return_value = Mock()

        tool = create_deeprag_tool(resource_config_dynamic, mock_llm)

        attachment_id = str(uuid.uuid4())
        mock_attachment = MockAttachment(
            ID=attachment_id, FullName="test.txt", MimeType="text/plain"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(
            attachment=mock_attachment, query="What is the summary?"
        )

        assert result == {"content": "Dynamic query result"}

        assert mock_interrupt.call_count == 1
        create_payload = mock_interrupt.call_args.args[0]
        assert isinstance(create_payload, CreateDeepRag)
        assert create_payload.attachments == [attachment_id]
        assert create_payload.prompt == "What is the summary?"
        assert create_payload.citation_mode == CitationMode.SKIP

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

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.deeprag_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch.dict(os.environ, {"UIPATH_FOLDER_KEY": "test-folder-key"})
    async def test_create_deeprag_passes_folder_key(
        self,
        mock_interrupt,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """CreateDeepRag.index_folder_key resolves from UIPATH_FOLDER_KEY at invoke time."""
        mock_interrupt.side_effect = [{"text": "result"}]
        mock_get_wrapper.return_value = Mock()

        tool = create_deeprag_tool(resource_config_static, mock_llm)
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="test.pdf", MimeType="application/pdf"
        )

        assert tool.coroutine is not None
        await tool.coroutine(attachment=mock_attachment)

        create_payload = mock_interrupt.call_args.args[0]
        assert isinstance(create_payload, CreateDeepRag)
        assert create_payload.index_folder_key == "test-folder-key"
