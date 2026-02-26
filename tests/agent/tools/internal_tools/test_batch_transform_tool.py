"""Tests for batch_transform_tool.py module."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field
from uipath.agent.models.agent import (
    AgentContextOutputColumn,
    AgentContextQuerySetting,
    AgentInternalBatchTransformSettings,
    AgentInternalBatchTransformToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
    BatchTransformFileExtension,
    BatchTransformFileExtensionSetting,
    BatchTransformWebSearchGrounding,
    BatchTransformWebSearchGroundingSetting,
)
from uipath.platform.context_grounding.context_grounding_index import (
    ContextGroundingIndex,
)

from uipath_langchain.agent.tools.internal_tools.batch_transform_tool import (
    create_batch_transform_tool,
)


class MockAttachment(BaseModel):
    """Mock attachment model for testing."""

    model_config = ConfigDict(populate_by_name=True)

    ID: str = Field(alias="ID")
    FullName: str = Field(alias="FullName")
    MimeType: str = Field(alias="MimeType")


class TestCreateBatchTransformTool:
    """Test cases for create_batch_transform_tool function."""

    @pytest.fixture
    def mock_llm(self):
        """Fixture for mock LLM."""
        return AsyncMock()

    @pytest.fixture
    def batch_transform_settings_static_query(self):
        """Fixture for Batch Transform settings with static query."""
        return AgentInternalBatchTransformSettings(
            context_type="attachment",
            query=AgentContextQuerySetting(
                value="Extract customer data", variant="static"
            ),
            folder_path_prefix=AgentContextQuerySetting(value="data/"),
            file_extension=BatchTransformFileExtensionSetting(
                value=BatchTransformFileExtension.CSV
            ),
            output_columns=[
                AgentContextOutputColumn(
                    name="customer_name", description="Name of the customer"
                ),
                AgentContextOutputColumn(
                    name="email", description="Customer email address"
                ),
            ],
            web_search_grounding=BatchTransformWebSearchGroundingSetting(
                value=BatchTransformWebSearchGrounding.DISABLED
            ),
        )

    @pytest.fixture
    def batch_transform_settings_dynamic_query(self):
        """Fixture for Batch Transform settings with dynamic query."""
        return AgentInternalBatchTransformSettings(
            context_type="attachment",
            query=AgentContextQuerySetting(
                description="Enter transformation query", variant="dynamic"
            ),
            folder_path_prefix=None,
            file_extension=BatchTransformFileExtensionSetting(
                value=BatchTransformFileExtension.CSV
            ),
            output_columns=[
                AgentContextOutputColumn(
                    name="result", description="Transformation result"
                ),
            ],
            web_search_grounding=BatchTransformWebSearchGroundingSetting(
                value=BatchTransformWebSearchGrounding.ENABLED
            ),
        )

    @pytest.fixture
    def resource_config_static(self, batch_transform_settings_static_query):
        """Fixture for resource configuration with static query."""
        input_schema = {
            "type": "object",
            "properties": {"attachment": {"type": "object"}},
            "required": ["attachment"],
        }
        output_schema = {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
        }

        properties = AgentInternalBatchTransformToolProperties(
            tool_type=AgentInternalToolType.BATCH_TRANSFORM,
            settings=batch_transform_settings_static_query,
        )

        return AgentInternalToolResourceConfig(
            name="batch_transform_static",
            description="Transform CSV with Batch Transform (static query)",
            input_schema=input_schema,
            output_schema=output_schema,
            properties=properties,
        )

    @pytest.fixture
    def resource_config_dynamic(self, batch_transform_settings_dynamic_query):
        """Fixture for resource configuration with dynamic query."""
        input_schema = {
            "type": "object",
            "properties": {"attachment": {"type": "object"}},
            "required": ["attachment"],
        }
        output_schema = {
            "type": "object",
            "properties": {"output": {"type": "string"}},
        }

        properties = AgentInternalBatchTransformToolProperties(
            tool_type=AgentInternalToolType.BATCH_TRANSFORM,
            settings=batch_transform_settings_dynamic_query,
        )

        return AgentInternalToolResourceConfig(
            name="batch_transform_dynamic",
            description="Transform CSV with Batch Transform (dynamic query)",
            input_schema=input_schema,
            output_schema=output_schema,
            properties=properties,
        )

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPathConfig"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_batch_transform_tool_static_query_index_ready(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_uipath_config,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test Batch Transform tool with static query when index is immediately ready."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath
        mock_uipath_config.job_key = "test-job-key"

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-batch-123",
            last_ingestion_status="Successful",
        )

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        # Index is ready → ReadyEphemeralIndex skips interrupt(). Only create_batch_transform fires.
        mock_interrupt.side_effect = [
            {"file_path": "/path/to/output.csv"},
        ]

        mock_attachment_uuid = uuid.uuid4()
        mock_uipath.jobs.create_attachment_async = AsyncMock(
            return_value=mock_attachment_uuid
        )

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_batch_transform_tool(resource_config_static, mock_llm)

        # Verify tool creation
        assert tool.name == "batch_transform_static"
        assert tool.description == "Transform CSV with Batch Transform (static query)"

        # Test tool execution
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="data.csv", MimeType="text/csv"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(attachment=mock_attachment)

        # Verify result contains attachment info
        assert result == {
            "result": {
                "ID": str(mock_attachment_uuid),
                "FullName": "output.csv",
                "MimeType": "text/csv",
            }
        }

        # Verify ephemeral index was created
        mock_uipath.context_grounding.create_ephemeral_index_async.assert_called_once()
        call_kwargs = (
            mock_uipath.context_grounding.create_ephemeral_index_async.call_args.kwargs
        )
        assert call_kwargs["usage"] == "BatchRAG"
        assert mock_attachment.ID in call_kwargs["attachments"]

        # Only create_batch_transform calls interrupt(); index was instant-resumed
        assert mock_interrupt.call_count == 1

        # Verify attachment was uploaded
        mock_uipath.jobs.create_attachment_async.assert_called_once_with(
            name="output.csv",
            source_path="output.csv",
            job_key="test-job-key",
        )

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPathConfig"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_batch_transform_tool_static_query_wait_for_ingestion(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_uipath_config,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test Batch Transform tool with static query when index needs to wait for ingestion."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath
        mock_uipath_config.job_key = "test-job-key"

        pending_id = str(uuid.uuid4())
        mock_index_pending = ContextGroundingIndex(
            id=pending_id,
            name="ephemeral-batch-456",
            last_ingestion_status="Queued",
        )

        mock_index_complete = {
            "id": mock_index_pending.id,
            "name": mock_index_pending.name,
            "last_ingestion_status": "Successful",
        }

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index_pending
        )

        # First interrupt returns completed index, second returns Batch Transform result
        mock_interrupt.side_effect = [
            mock_index_complete,
            {"file_path": "/path/to/transformed.csv"},
        ]

        mock_attachment_uuid = uuid.uuid4()
        mock_uipath.jobs.create_attachment_async = AsyncMock(
            return_value=mock_attachment_uuid
        )

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_batch_transform_tool(resource_config_static, mock_llm)

        # Test tool execution
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="data.csv", MimeType="text/csv"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(attachment=mock_attachment)

        # Verify result contains attachment info
        assert result == {
            "result": {
                "ID": str(mock_attachment_uuid),
                "FullName": "output.csv",
                "MimeType": "text/csv",
            }
        }

        # Verify interrupt was called twice (WaitEphemeralIndex + CreateBatchTransform)
        assert mock_interrupt.call_count == 2

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPathConfig"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_batch_transform_tool_dynamic_query(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_uipath_config,
        mock_get_wrapper,
        resource_config_dynamic,
        mock_llm,
    ):
        """Test Batch Transform tool with dynamic query."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath
        mock_uipath_config.job_key = "test-job-key"

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-batch-789",
            last_ingestion_status="Successful",
        )

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        # Index is ready → ReadyEphemeralIndex skips interrupt(). Only create_batch_transform fires.
        mock_interrupt.side_effect = [
            {"output": "Transformation complete"},
        ]

        mock_attachment_uuid = uuid.uuid4()
        mock_uipath.jobs.create_attachment_async = AsyncMock(
            return_value=mock_attachment_uuid
        )

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_batch_transform_tool(resource_config_dynamic, mock_llm)

        # Test tool execution with dynamic query
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="input.csv", MimeType="text/csv"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(
            attachment=mock_attachment, query="Extract all names"
        )

        # Verify result contains attachment info
        assert result == {
            "result": {
                "ID": str(mock_attachment_uuid),
                "FullName": "output.csv",
                "MimeType": "text/csv",
            }
        }

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPathConfig"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_batch_transform_tool_default_destination_path(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_uipath_config,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test Batch Transform tool defaults to output.csv when destination_path not provided."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath
        mock_uipath_config.job_key = "test-job-key"

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-batch-default",
            last_ingestion_status="Successful",
        )

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        # Index is ready → ReadyEphemeralIndex skips interrupt(). Only create_batch_transform fires.
        mock_interrupt.side_effect = [
            {"file_path": "output.csv"},
        ]

        mock_attachment_uuid = uuid.uuid4()
        mock_uipath.jobs.create_attachment_async = AsyncMock(
            return_value=mock_attachment_uuid
        )

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_batch_transform_tool(resource_config_static, mock_llm)

        # Test tool execution without destination_path
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="data.csv", MimeType="text/csv"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(attachment=mock_attachment)

        # Verify result contains attachment info with default destination_path
        assert result == {
            "result": {
                "ID": str(mock_attachment_uuid),
                "FullName": "output.csv",
                "MimeType": "text/csv",
            }
        }

        # Only create_batch_transform calls interrupt(); index was instant-resumed
        assert mock_interrupt.call_count == 1

        # Verify attachment was uploaded with default path
        mock_uipath.jobs.create_attachment_async.assert_called_once_with(
            name="output.csv",
            source_path="output.csv",
            job_key="test-job-key",
        )

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPathConfig"
    )
    @patch("uipath_langchain.agent.tools.internal_tools.batch_transform_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    @patch(
        "uipath_langchain.agent.tools.internal_tools.batch_transform_tool.mockable",
        lambda **kwargs: lambda f: f,
    )
    async def test_create_batch_transform_tool_custom_destination_path(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_uipath_config,
        mock_get_wrapper,
        resource_config_static,
        mock_llm,
    ):
        """Test Batch Transform tool with custom destination_path."""
        # Setup mocks
        mock_uipath = AsyncMock()
        mock_uipath_class.return_value = mock_uipath
        mock_uipath_config.job_key = "test-job-key"

        mock_index = ContextGroundingIndex(
            id=str(uuid.uuid4()),
            name="ephemeral-batch-custom",
            last_ingestion_status="Successful",
        )

        mock_uipath.context_grounding.create_ephemeral_index_async = AsyncMock(
            return_value=mock_index
        )

        # Index is ready → ReadyEphemeralIndex skips interrupt(). Only create_batch_transform fires.
        mock_interrupt.side_effect = [
            {"file_path": "/custom/path/result.csv"},
        ]

        mock_attachment_uuid = uuid.uuid4()
        mock_uipath.jobs.create_attachment_async = AsyncMock(
            return_value=mock_attachment_uuid
        )

        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        # Create tool
        tool = create_batch_transform_tool(resource_config_static, mock_llm)

        # Test tool execution with custom destination_path
        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="data.csv", MimeType="text/csv"
        )

        assert tool.coroutine is not None
        result = await tool.coroutine(
            attachment=mock_attachment, destination_path="/custom/path/result.csv"
        )

        # Verify result contains attachment info with custom path
        assert result == {
            "result": {
                "ID": str(mock_attachment_uuid),
                "FullName": "/custom/path/result.csv",
                "MimeType": "text/csv",
            }
        }

        # Verify attachment was uploaded with custom path
        mock_uipath.jobs.create_attachment_async.assert_called_once_with(
            name="/custom/path/result.csv",
            source_path="/custom/path/result.csv",
            job_key="test-job-key",
        )

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    async def test_create_batch_transform_tool_missing_attachment(
        self, mock_get_wrapper, resource_config_static, mock_llm
    ):
        """Test tool execution fails when attachment is missing."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_batch_transform_tool(resource_config_static, mock_llm)

        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="Argument 'attachment' is not available"):
            await tool.coroutine()

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    async def test_create_batch_transform_tool_missing_query_dynamic(
        self, mock_get_wrapper, resource_config_dynamic, mock_llm
    ):
        """Test tool execution fails when query is missing (dynamic mode)."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_batch_transform_tool(resource_config_dynamic, mock_llm)

        mock_attachment = MockAttachment(
            ID=str(uuid.uuid4()), FullName="data.csv", MimeType="text/csv"
        )

        assert tool.coroutine is not None
        with pytest.raises(
            ValueError, match="Query is required for Batch Transform tool"
        ):
            await tool.coroutine(attachment=mock_attachment)

    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_wrapper"
    )
    async def test_create_batch_transform_tool_missing_attachment_id(
        self, mock_get_wrapper, resource_config_static, mock_llm
    ):
        """Test tool execution fails when attachment ID is missing."""
        mock_wrapper = Mock()
        mock_get_wrapper.return_value = mock_wrapper

        tool = create_batch_transform_tool(resource_config_static, mock_llm)

        class AttachmentWithoutID(BaseModel):
            FullName: str
            MimeType: str

        mock_attachment = AttachmentWithoutID(FullName="data.csv", MimeType="text/csv")

        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="Attachment ID is required"):
            await tool.coroutine(attachment=mock_attachment)
