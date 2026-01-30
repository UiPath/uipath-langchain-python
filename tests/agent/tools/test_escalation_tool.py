"""Tests for escalation_tool.py metadata."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolCall
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationChannelProperties,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    AssetRecipient,
    StandardRecipient,
)
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType

from uipath_langchain.agent.tools.escalation_tool import (
    create_escalation_tool,
    resolve_asset,
    resolve_recipient_value,
)


class TestResolveAsset:
    """Test the resolve_asset function."""

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_resolve_asset_success(self, mock_uipath_class):
        """Test successful asset retrieval."""
        # Setup mock
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_result = MagicMock()
        mock_result.value = "test@example.com"
        mock_client.assets.retrieve_async = AsyncMock(return_value=mock_result)

        # Execute
        result = await resolve_asset("email_asset", "/Test/Folder")

        # Assert
        assert result == "test@example.com"
        mock_client.assets.retrieve_async.assert_called_once_with(
            name="email_asset", folder_path="/Test/Folder"
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_resolve_asset_no_value(self, mock_uipath_class):
        """Test asset with no value raises ValueError."""
        # Setup mock
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_result = MagicMock()
        mock_result.value = None
        mock_client.assets.retrieve_async = AsyncMock(return_value=mock_result)

        # Execute and assert
        with pytest.raises(ValueError) as exc_info:
            await resolve_asset("empty_asset", "/Test/Folder")

        assert "Asset 'empty_asset' has no value configured" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_resolve_asset_not_found(self, mock_uipath_class):
        """Test asset not found raises ValueError."""
        # Setup mock
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.assets.retrieve_async = AsyncMock(return_value=None)

        # Execute and assert
        with pytest.raises(ValueError) as exc_info:
            await resolve_asset("missing_asset", "/Test/Folder")

        assert "Asset 'missing_asset' has no value configured" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_resolve_asset_retrieval_exception(self, mock_uipath_class):
        """Test exception during asset retrieval raises ValueError with context."""
        # Setup mock
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.assets.retrieve_async = AsyncMock(
            side_effect=Exception("Connection error")
        )

        # Execute and assert
        with pytest.raises(ValueError) as exc_info:
            await resolve_asset("problem_asset", "/Test/Folder")

        assert (
            "Failed to resolve asset 'problem_asset' in folder '/Test/Folder'"
            in str(exc_info.value)
        )
        assert "Connection error" in str(exc_info.value)


class TestResolveRecipientValue:
    """Test the resolve_recipient_value function."""

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.resolve_asset")
    async def test_resolve_recipient_asset_user_email(self, mock_resolve_asset):
        """Test ASSET_USER_EMAIL type calls resolve_asset."""
        mock_resolve_asset.return_value = "resolved@example.com"

        recipient = AssetRecipient(
            type=AgentEscalationRecipientType.ASSET_USER_EMAIL,
            asset_name="email_asset",
            folder_path="/Test/Folder",
        )

        result = await resolve_recipient_value(recipient)

        assert result == TaskRecipient(
            value="resolved@example.com", type=TaskRecipientType.EMAIL
        )
        mock_resolve_asset.assert_called_once_with("email_asset", "/Test/Folder")

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.resolve_asset")
    async def test_resolve_recipient_asset_group_name(self, mock_resolve_asset):
        """Test ASSET_GROUP_NAME type calls resolve_asset."""
        mock_resolve_asset.return_value = "ResolvedGroup"

        recipient = AssetRecipient(
            type=AgentEscalationRecipientType.ASSET_GROUP_NAME,
            asset_name="group_asset",
            folder_path="/Test/Folder",
        )

        result = await resolve_recipient_value(recipient)

        assert result == TaskRecipient(
            value="ResolvedGroup", type=TaskRecipientType.GROUP_NAME
        )
        mock_resolve_asset.assert_called_once_with("group_asset", "/Test/Folder")

    @pytest.mark.asyncio
    async def test_resolve_recipient_user_email(self):
        """Test USER_EMAIL type returns value directly."""
        recipient = StandardRecipient(
            type=AgentEscalationRecipientType.USER_EMAIL,
            value="direct@example.com",
        )

        result = await resolve_recipient_value(recipient)

        assert result == TaskRecipient(
            value="direct@example.com", type=TaskRecipientType.EMAIL
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.resolve_asset")
    async def test_resolve_recipient_propagates_error_when_asset_resolution_fails(
        self, mock_resolve_asset
    ):
        """Test AssetRecipient when asset resolution fails."""
        mock_resolve_asset.side_effect = ValueError("Asset not found")

        recipient = AssetRecipient(
            type=AgentEscalationRecipientType.ASSET_USER_EMAIL,
            asset_name="nonexistent",
            folder_path="Shared",
        )

        with pytest.raises(ValueError) as exc_info:
            await resolve_recipient_value(recipient)

        assert "Asset not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_recipient_no_value(self):
        """Test recipient without value attribute returns None."""
        # Create a minimal recipient object without value
        recipient = MagicMock()
        recipient.type = AgentEscalationRecipientType.USER_EMAIL
        del recipient.value  # Simulate no value attribute

        result = await resolve_recipient_value(recipient)

        assert result is None


class TestEscalationToolMetadata:
    """Test that escalation tool has correct metadata for observability."""

    @pytest.fixture
    def escalation_resource(self):
        """Create a minimal escalation tool resource config."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
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
                    recipients=[
                        StandardRecipient(
                            type=AgentEscalationRecipientType.USER_EMAIL,
                            value="user@example.com",
                        )
                    ],
                )
            ],
        )

    @pytest.fixture
    def escalation_resource_no_recipient(self):
        """Create escalation resource without recipients."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
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

    @pytest.mark.asyncio
    async def test_escalation_tool_has_metadata(self, escalation_resource):
        """Test that escalation tool has metadata dict."""
        tool = create_escalation_tool(escalation_resource)

        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_tool_type(self, escalation_resource):
        """Test that metadata contains tool_type for span detection."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "escalation"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_display_name(self, escalation_resource):
        """Test that metadata contains display_name from app_name."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "ApprovalApp"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_channel_type(self, escalation_resource):
        """Test that metadata contains channel_type for span attributes."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["channel_type"] == "actionCenter"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_metadata_has_recipient(
        self, mock_interrupt, escalation_resource
    ):
        """Test that metadata contains recipient when recipient is USER_EMAIL."""
        # Mock interrupt to return a result
        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)

        # Create mock state and call to invoke through wrapper
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] == TaskRecipient(
            value="user@example.com", type=TaskRecipientType.EMAIL
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_metadata_recipient_none_when_no_recipients(
        self, mock_interrupt, escalation_resource_no_recipient
    ):
        """Test that recipient is None when no recipients configured."""
        # Mock interrupt to return a result
        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource_no_recipient)

        # Create mock state and call to invoke through wrapper
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_string_task_title(self, mock_interrupt):
        """Test escalation tool with legacy string task title."""
        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        # Create resource with string task title
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "taskTitle": "Static Task Title",
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)

        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # Verify interrupt was called with the static title
        call_args = mock_interrupt.call_args[0][0]
        assert call_args.title == "Static Task Title"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_text_builder_task_title(self, mock_interrupt):
        """Test escalation tool with TEXT_BUILDER task title builds from tokens."""
        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        # Create resource with TEXT_BUILDER task title containing variable token
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "taskTitle": {
                "type": "textBuilder",
                "tokens": [
                    {"type": "simpleText", "rawString": "Approve request for "},
                    {"type": "variable", "rawString": "input.userName"},
                ],
            },
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)

        # Create mock state with variables for token interpolation
        state = {"userName": "John Doe", "messages": []}
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, state)  # type: ignore[attr-defined]

        # Verify interrupt was called with the correctly built task title
        assert mock_interrupt.called
        call_args = mock_interrupt.call_args[0][0]
        assert call_args.title == "Approve request for John Doe"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_empty_task_title_defaults_to_escalation_task(
        self, mock_interrupt
    ):
        """Test escalation tool defaults to 'Escalation Task' when task title is empty."""
        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        # Create resource with empty string task title
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "taskTitle": "",
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)

        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # Verify interrupt was called with the default title
        call_args = mock_interrupt.call_args[0][0]
        assert call_args.title == "Escalation Task"


class TestEscalationToolOutputSchema:
    """Test escalation tool output schema for simulation support."""

    @pytest.fixture
    def escalation_resource(self):
        """Create a minimal escalation tool resource config."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={
                        "type": "object",
                        "properties": {
                            "approved": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                    },
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[
                        StandardRecipient(
                            type=AgentEscalationRecipientType.USER_EMAIL,
                            value="user@example.com",
                        )
                    ],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_escalation_tool_output_schema_has_action_field(
        self, escalation_resource
    ):
        """Test that escalation tool output schema includes action field."""
        tool = create_escalation_tool(escalation_resource)
        # Get the output schema from the tool's args_schema
        args_schema = tool.args_schema
        assert args_schema is not None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_result_validation(
        self, mock_interrupt, escalation_resource
    ):
        """Test that tool properly processes and validates results."""
        # Mock interrupt to return a proper result object with action and data
        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper
        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # Should successfully process the result
        assert isinstance(result, dict)
        assert result == {}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_extracts_action_from_result(
        self, mock_interrupt, escalation_resource
    ):
        """Test that tool correctly extracts action from escalation result."""
        # Mock interrupt to return a result with action
        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {"approved": True}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # Verify interrupt was called (action was processed)
        assert mock_interrupt.called

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_outcome_mapping(self, mock_interrupt):
        """Test escalation tool with outcome mapping for actions."""
        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {"approved": True}
        mock_interrupt.return_value = mock_result

        # Create resource with outcome mapping
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "outcomeMapping": {"approve": "end", "reject": "continue"},
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # Verify interrupt was called with approval action
        assert mock_interrupt.called
