"""Tests for ixp_escalation_tool.py functionality."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolCall
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationChannelProperties,
    AgentIxpVsEscalationResourceConfig,
)
from uipath.platform.common import WaitDocumentExtractionValidation
from uipath.platform.documents import (
    ActionPriority,
    ExtractionResponseIXP,
    ValidateExtractionAction,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.react.types import (
    AgentGraphState,
    InnerAgentGraphState,
)
from uipath_langchain.agent.tools.ixp_escalation_tool import (
    create_ixp_escalation_tool,
)


def _passthrough_task(fn):
    """Replace @task decorator with a passthrough for tests."""
    return fn


def _make_mock_validation_response(
    operation_id: str = "test-op-id",
    task_url: str = "https://example.com/actions_/tasks/123",
) -> MagicMock:
    """Create a mock ValidateExtractionAction response from the SDK."""
    mock = MagicMock(spec=ValidateExtractionAction)
    mock.operation_id = operation_id
    mock.action_data = {"taskUrl": task_url}
    return mock


def _make_extraction_response_dict() -> dict[str, Any]:
    """Create an extraction response dict as stored in tools_storage."""
    return ExtractionResponseIXP(
        project_id="test-project",
        tag="v1.0",
        extraction_result={
            "DocumentId": "doc-123",
            "ResultsVersion": 1,
            "ResultsDocument": {},
        },
        project_type="IXP",
        document_type_id="doc-type-123",
        data_projection=[],
    ).model_dump(by_alias=True)


def _make_interrupt_response(rejected: bool = False) -> dict[str, Any]:
    """Create a mock interrupt resume response."""
    data: dict[str, Any] = {}
    if rejected:
        data["documentRejectionDetails"] = {"reason": "Invalid document"}
    return {
        "dataProjection": [],
        "actionData": {"data": data},
    }


class TestIxpEscalationToolCreation:
    """Test that ixp escalation tool is created correctly."""

    @pytest.fixture
    def escalation_resource(self):
        """Create a minimal ixp escalation tool resource config."""
        return AgentIxpVsEscalationResourceConfig(
            name="validate_invoice",
            description="Validate extracted invoice data",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "some_tool_id",
                "storageBucketName": "some_bucket_name",
                "storageBucketFolderPath": "some_solution_folder",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

    def test_tool_has_correct_name(self, escalation_resource):
        """Test that tool has sanitized name."""
        tool = create_ixp_escalation_tool(escalation_resource)
        assert tool.name == "validate_invoice"

    def test_tool_has_correct_description(self, escalation_resource):
        """Test that tool has correct description."""
        tool = create_ixp_escalation_tool(escalation_resource)
        assert tool.description == "Validate extracted invoice data"

    def test_tool_has_wrapper_configured(self, escalation_resource):
        """Test that tool has wrapper configured."""
        tool = create_ixp_escalation_tool(escalation_resource)
        assert hasattr(tool, "awrapper")
        assert tool.awrapper is not None

    def test_tool_uses_empty_input_schema(self, escalation_resource):
        """Test that tool uses empty dict as input schema."""
        tool = create_ixp_escalation_tool(escalation_resource)
        assert tool.args_schema == {}

    def test_tool_has_vs_escalation_metadata(self, escalation_resource):
        """Test that tool has vs_escalation metadata with ixp_tool_id and storage_bucket_name."""
        tool = create_ixp_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "vs_escalation"
        assert tool.metadata["ixp_tool_id"] == "some_tool_id"
        assert tool.metadata["storage_bucket_name"] == "some_bucket_name"


class TestIxpEscalationToolWrapper:
    """Test the wrapper functionality of ixp escalation tool."""

    @pytest.fixture
    def escalation_resource(self):
        """Create ixp escalation tool resource config."""
        return AgentIxpVsEscalationResourceConfig(
            name="validate_data",
            description="Validate extracted data",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "data_extraction",
                "storageBucketName": "validation-bucket",
                "storageBucketFolderPath": "/validations",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ValidationApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                    task_title="Data Validation",
                    priority="Medium",
                )
            ],
        )

    @pytest.fixture
    def mock_state_with_extraction(self):
        """Create a mock state with extraction result."""
        return AgentGraphState(
            messages=[],
            inner_state=InnerAgentGraphState(
                tools_storage={"data_extraction": _make_extraction_response_dict()}
            ),
        )

    @pytest.fixture
    def mock_state_without_extraction(self):
        """Create a mock state without extraction result."""
        return AgentGraphState(
            messages=[], inner_state=InnerAgentGraphState(tools_storage={})
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.ixp_escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    async def test_wrapper_retrieves_extraction_from_state(
        self,
        mock_interrupt,
        mock_uipath_cls,
        escalation_resource,
        mock_state_with_extraction,
    ):
        """Test that wrapper retrieves extraction result from state."""
        mock_validation_response = _make_mock_validation_response()
        mock_client = MagicMock()
        mock_client.documents.create_validate_extraction_action_async = AsyncMock(
            return_value=mock_validation_response
        )
        mock_uipath_cls.return_value = mock_client
        mock_interrupt.return_value = _make_interrupt_response()

        tool = create_ixp_escalation_tool(escalation_resource)
        call = ToolCall(id="call-1", name="validate_data", args={})

        assert hasattr(tool, "awrapper")
        await tool.awrapper(tool, call, mock_state_with_extraction)

        # SDK was called with the extraction result from state
        mock_client.documents.create_validate_extraction_action_async.assert_called_once()
        sdk_kwargs = (
            mock_client.documents.create_validate_extraction_action_async.call_args
        )
        assert isinstance(
            sdk_kwargs.kwargs["extraction_response"], ExtractionResponseIXP
        )

        # interrupt was called with WaitDocumentExtractionValidation including task_url
        assert mock_interrupt.called
        validation_arg = mock_interrupt.call_args[0][0]
        assert isinstance(validation_arg, WaitDocumentExtractionValidation)
        assert validation_arg.extraction_validation.operation_id == "test-op-id"
        assert validation_arg.task_url == "https://example.com/actions_/tasks/123"

    @pytest.mark.asyncio
    async def test_wrapper_raises_error_when_extraction_not_found(
        self, escalation_resource, mock_state_without_extraction
    ):
        """Test that wrapper raises RuntimeError when extraction result not found."""
        tool = create_ixp_escalation_tool(escalation_resource)
        call = ToolCall(id="call-1", name="validate_data", args={})

        with pytest.raises(RuntimeError) as exc_info:
            assert hasattr(tool, "awrapper")
            await tool.awrapper(tool, call, mock_state_without_extraction)

        assert "Extraction result not found for data_extraction" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wrapper_looks_for_correct_ixp_tool_id(
        self, mock_state_with_extraction
    ):
        """Test that wrapper uses correct ixp_tool_id from config."""
        # tool with different ixp_tool_id
        resource = AgentIxpVsEscalationResourceConfig(
            name="validate",
            description="Validate",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "different_extraction",
                "storageBucketName": "bucket",
                "storageBucketFolderPath": "/path",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="TestApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

        tool = create_ixp_escalation_tool(resource)
        call = ToolCall(id="call-1", name="validate", args={})

        # "different_extraction" is not in state
        with pytest.raises(RuntimeError) as exc_info:
            assert hasattr(tool, "awrapper")
            await tool.awrapper(tool, call, mock_state_with_extraction)

        assert "Extraction result not found for different_extraction" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.ixp_escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    async def test_wrapper_raises_on_document_rejection(
        self,
        mock_interrupt,
        mock_uipath_cls,
        escalation_resource,
        mock_state_with_extraction,
    ):
        """Test that wrapper raises AgentRuntimeError on document rejection."""
        mock_validation_response = _make_mock_validation_response()
        mock_client = MagicMock()
        mock_client.documents.create_validate_extraction_action_async = AsyncMock(
            return_value=mock_validation_response
        )
        mock_uipath_cls.return_value = mock_client
        mock_interrupt.return_value = _make_interrupt_response(rejected=True)

        tool = create_ixp_escalation_tool(escalation_resource)
        call = ToolCall(id="call-1", name="validate_data", args={})

        with pytest.raises(AgentRuntimeError) as exc_info:
            assert hasattr(tool, "awrapper")
            await tool.awrapper(tool, call, mock_state_with_extraction)

        assert "TERMINATION_ESCALATION_REJECTED" in exc_info.value.error_info.code
        assert "Invalid document" in exc_info.value.error_info.detail


class TestIxpEscalationToolExecution:
    """Test the execution of ixp escalation tool."""

    @pytest.fixture
    def escalation_resource(self):
        """Create ixp escalation tool resource config."""
        return AgentIxpVsEscalationResourceConfig(
            name="validate_invoice",
            description="Validate invoice",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "invoice_extraction",
                "storageBucketName": "invoices-bucket",
                "storageBucketFolderPath": "/validations",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="InvoiceApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                    task_title="Invoice Validation",
                    priority="High",
                )
            ],
        )

    @pytest.fixture
    def mock_extraction_response(self):
        """Create a mock extraction response."""
        return ExtractionResponseIXP(
            project_id="invoice-project",
            tag="v2.0",
            extraction_result={
                "DocumentId": "invoice-doc-456",
                "ResultsVersion": 1,
                "ResultsDocument": {},
            },
            project_type="IXP",
            document_type_id="invoice-doc-type",
            data_projection=[],
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.ixp_escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    async def test_tool_calls_interrupt_with_correct_params(
        self,
        mock_interrupt,
        mock_uipath_cls,
        escalation_resource,
        mock_extraction_response,
    ):
        """Test that tool calls SDK with correct params and interrupts with WaitDocumentExtractionValidation."""
        mock_validation_response = _make_mock_validation_response()
        mock_client = MagicMock()
        mock_client.documents.create_validate_extraction_action_async = AsyncMock(
            return_value=mock_validation_response
        )
        mock_uipath_cls.return_value = mock_client
        mock_interrupt.return_value = _make_interrupt_response()

        tool = create_ixp_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        tool.metadata["extraction_result"] = mock_extraction_response

        await tool.ainvoke({})

        # SDK was called with correct parameters
        mock_client.documents.create_validate_extraction_action_async.assert_called_once_with(
            extraction_response=mock_extraction_response,
            action_title="VS Escalation Task",
            storage_bucket_name="invoices-bucket",
            storage_bucket_directory_path="/validations",
            action_priority=ActionPriority.HIGH,
        )

        # interrupt was called with WaitDocumentExtractionValidation
        assert mock_interrupt.called
        validation_arg = mock_interrupt.call_args[0][0]
        assert isinstance(validation_arg, WaitDocumentExtractionValidation)
        assert validation_arg.extraction_validation.operation_id == "test-op-id"
        assert validation_arg.task_url == "https://example.com/actions_/tasks/123"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.ixp_escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    async def test_tool_uses_default_action_title_when_not_provided(
        self, mock_interrupt, mock_uipath_cls, mock_extraction_response
    ):
        """Test that tool uses default action title when not provided."""
        resource = AgentIxpVsEscalationResourceConfig(
            name="validate",
            description="Validate",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "extraction",
                "storageBucketName": "bucket",
                "storageBucketFolderPath": "/path",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="TestApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

        mock_validation_response = _make_mock_validation_response()
        mock_client = MagicMock()
        mock_client.documents.create_validate_extraction_action_async = AsyncMock(
            return_value=mock_validation_response
        )
        mock_uipath_cls.return_value = mock_client
        mock_interrupt.return_value = _make_interrupt_response()

        tool = create_ixp_escalation_tool(resource)
        assert tool.metadata is not None
        tool.metadata["extraction_result"] = mock_extraction_response

        await tool.ainvoke({})

        sdk_kwargs = (
            mock_client.documents.create_validate_extraction_action_async.call_args
        )
        assert sdk_kwargs.kwargs["action_title"] == "VS Escalation Task"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.ixp_escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    async def test_tool_uses_default_priority_when_not_provided(
        self, mock_interrupt, mock_uipath_cls, mock_extraction_response
    ):
        """Test that tool uses default priority when not provided."""
        resource = AgentIxpVsEscalationResourceConfig(
            name="validate",
            description="Validate",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "extraction",
                "storageBucketName": "bucket",
                "storageBucketFolderPath": "/path",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="TestApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

        mock_validation_response = _make_mock_validation_response()
        mock_client = MagicMock()
        mock_client.documents.create_validate_extraction_action_async = AsyncMock(
            return_value=mock_validation_response
        )
        mock_uipath_cls.return_value = mock_client
        mock_interrupt.return_value = _make_interrupt_response()

        tool = create_ixp_escalation_tool(resource)
        assert tool.metadata is not None
        tool.metadata["extraction_result"] = mock_extraction_response

        await tool.ainvoke({})

        sdk_kwargs = (
            mock_client.documents.create_validate_extraction_action_async.call_args
        )
        assert sdk_kwargs.kwargs["action_priority"] == ActionPriority.MEDIUM

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.ixp_escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    async def test_tool_returns_data_projection_as_dict(
        self,
        mock_interrupt,
        mock_uipath_cls,
        escalation_resource,
        mock_extraction_response,
    ):
        """Test that tool returns data projection as a dict."""
        mock_validation_response = _make_mock_validation_response()
        mock_client = MagicMock()
        mock_client.documents.create_validate_extraction_action_async = AsyncMock(
            return_value=mock_validation_response
        )
        mock_uipath_cls.return_value = mock_client
        mock_interrupt.return_value = _make_interrupt_response()

        tool = create_ixp_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        tool.metadata["extraction_result"] = mock_extraction_response

        result = await tool.ainvoke({})

        assert isinstance(result, dict)
        assert "data" in result

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.ixp_escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.durable_interrupt.interrupt")
    async def test_tool_stores_validation_response_in_metadata(
        self,
        mock_interrupt,
        mock_uipath_cls,
        escalation_resource,
        mock_extraction_response,
    ):
        """Test that tool stores full validation response in metadata for wrapper rejection check."""
        mock_validation_response = _make_mock_validation_response()
        mock_client = MagicMock()
        mock_client.documents.create_validate_extraction_action_async = AsyncMock(
            return_value=mock_validation_response
        )
        mock_uipath_cls.return_value = mock_client
        interrupt_response = _make_interrupt_response()
        mock_interrupt.return_value = interrupt_response

        tool = create_ixp_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        tool.metadata["extraction_result"] = mock_extraction_response

        await tool.ainvoke({})

        assert tool.metadata["_validation_response"] == interrupt_response


class TestIxpEscalationToolNameSanitization:
    """Test that tool names are properly sanitized."""

    @pytest.mark.asyncio
    async def test_tool_name_with_spaces(self):
        """Test that tool names with spaces are sanitized."""
        resource = AgentIxpVsEscalationResourceConfig(
            name="Validate Invoice Data",
            description="Validate",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "extraction",
                "storageBucketName": "bucket",
                "storageBucketFolderPath": "/path",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="TestApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

        tool = create_ixp_escalation_tool(resource)
        assert " " not in tool.name

    @pytest.mark.asyncio
    async def test_tool_name_with_special_chars(self):
        """Test that tool names with special characters are sanitized."""
        resource = AgentIxpVsEscalationResourceConfig(
            name="validate-data@v1",
            description="Validate",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            vs_escalation_properties={
                "ixpToolId": "extraction",
                "storageBucketName": "bucket",
                "storageBucketFolderPath": "/path",
            },
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="TestApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

        tool = create_ixp_escalation_tool(resource)
        assert tool.name is not None
        assert len(tool.name) > 0
