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
from uipath.platform.action_center.tasks import Task, TaskRecipient, TaskRecipientType

from uipath_langchain.agent.tools.escalation_tool import (
    _get_user_email,
    _parse_task_data,
    create_escalation_tool,
    resolve_asset,
    resolve_recipient_value,
)


def _make_mock_task(**overrides):
    """Create a Task instance for tests."""
    defaults = {"id": 1, "key": "task-key", "title": "Test Task"}
    defaults.update(overrides)
    return Task(**defaults)


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
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_metadata_has_recipient(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that metadata contains recipient when recipient is USER_EMAIL."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] == TaskRecipient(
            value="user@example.com", type=TaskRecipientType.EMAIL
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_metadata_recipient_none_when_no_recipients(
        self, mock_interrupt, mock_uipath_class, escalation_resource_no_recipient
    ):
        """Test that recipient is None when no recipients configured."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource_no_recipient)

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_string_task_title(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool with legacy string task title."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

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

        # Verify create_async was called with the static title
        create_call = mock_client.tasks.create_async.call_args
        assert create_call[1]["title"] == "Static Task Title"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_text_builder_task_title(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool with TEXT_BUILDER task title builds from tokens."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

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

        # Verify create_async was called with the correctly built task title
        create_call = mock_client.tasks.create_async.call_args
        assert create_call[1]["title"] == "Approve request for John Doe"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_empty_task_title_defaults_to_escalation_task(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool defaults to 'Escalation Task' when task title is empty."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

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

        # Verify create_async was called with the default title
        create_call = mock_client.tasks.create_async.call_args
        assert create_call[1]["title"] == "Escalation Task"


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
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_result_validation(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that tool properly processes and validates results."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 123
        mock_result.key = None
        mock_result.assigned_to_user = None
        mock_result.action = "approve"
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert isinstance(result, dict)
        assert result["outcome"] == "approve"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_extracts_action_from_result(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that tool correctly extracts action from escalation result."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {"approved": True}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert mock_interrupt.called

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_escalation_tool_with_outcome_mapping_end(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool with outcome mapping that ends agent."""
        from uipath_langchain.agent.exceptions import AgentTerminationException

        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 456
        mock_result.key = None
        mock_result.assigned_to_user = None
        mock_result.action = "approve"
        mock_result.data = {"approved": True}
        mock_interrupt.return_value = mock_result

        # Create resource with outcome mapping where approve -> end
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

        # Invoke through the wrapper - should raise AgentTerminationException
        with pytest.raises(AgentTerminationException):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert mock_interrupt.called


class TestGetUserEmail:
    """Test the _get_user_email helper function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert _get_user_email(None) is None

    def test_dict_with_email_address(self):
        """Test extraction from dict with emailAddress field."""
        user = {"emailAddress": "test@example.com", "name": "Test"}
        assert _get_user_email(user) == "test@example.com"

    def test_dict_without_email_address(self):
        """Test dict without emailAddress returns None."""
        user = {"name": "Test", "id": 123}
        assert _get_user_email(user) is None

    def test_object_with_email_address(self):
        """Test extraction from object with emailAddress attribute."""
        user = MagicMock(emailAddress="test@example.com")
        assert _get_user_email(user) == "test@example.com"

    def test_object_without_email_address(self):
        """Test object without emailAddress attribute returns None."""
        user = MagicMock(spec=["name", "id"])
        assert _get_user_email(user) is None


class TestEscalationToolTaskInfo:
    """Test that escalation tool extracts task_id and assigned_to."""

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
                    recipients=[],
                )
            ],
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_wrapper_returns_task_id_and_assigned_to(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that wrapper result includes task_id and assigned_to from Task."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 12345
        mock_result.key = None
        mock_result.assigned_to_user = {"emailAddress": "user@example.com"}
        mock_result.action = "approve"
        mock_result.data = {"reason": "looks good"}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["task_id"] == 12345
        assert result["assigned_to"] == "user@example.com"
        assert result["outcome"] == "approve"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_wrapper_handles_missing_assigned_to_user(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that wrapper handles None assigned_to_user gracefully."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 99999
        mock_result.key = None
        mock_result.assigned_to_user = None
        mock_result.action = "reject"
        mock_result.data = {}
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["task_id"] == 99999
        assert result["assigned_to"] is None


class TestEscalationToolCreatesTaskBeforeInterrupt:
    """Test that escalation tool creates task inline before calling interrupt."""

    @pytest.fixture
    def escalation_resource(self):
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
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.interrupt")
    async def test_creates_task_then_interrupts_with_wait_escalation(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test task is created via create_async, then interrupt(WaitEscalation)."""
        from uipath.platform.common import WaitEscalation

        task = _make_mock_task(id=555, key="task-key-555")
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 555
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.assigned_to_user = None
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={"field": "value"}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_client.tasks.create_async.assert_called_once()

        # Verify interrupt was called with WaitEscalation containing the task
        mock_interrupt.assert_called_once()
        interrupt_arg = mock_interrupt.call_args[0][0]
        assert isinstance(interrupt_arg, WaitEscalation)
        assert interrupt_arg.action == task

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_task_creation_failure_propagates(
        self, mock_uipath_class, escalation_resource
    ):
        """Test that task creation failure propagates as exception."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(side_effect=Exception("API error"))
        mock_uipath_class.return_value = mock_client

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(Exception, match="API error"):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]


class TestParseTaskData:
    """Test output task data is filtered correctly."""

    def test_filters_input_fields_when_no_output_schema(self):
        """Test that input fields are excluded when output_schema is None."""
        data = {"input_field": "value1", "output_field": "value2"}
        input_schema = {"properties": {"input_field": {"type": "string"}}}

        result = _parse_task_data(data, input_schema, output_schema=None)

        assert result == {"output_field": "value2"}
        assert "input_field" not in result

    def test_includes_only_output_fields_when_output_schema_provided(self):
        """Test that only output schema fields are included."""
        data = {"field1": "a", "field2": "b", "field3": "c"}
        input_schema = {"properties": {"field1": {"type": "string"}}}
        output_schema = {
            "properties": {"field1": {"type": "string"}, "field2": {"type": "string"}}
        }

        result = _parse_task_data(data, input_schema, output_schema)

        assert result == {"field1": "a", "field2": "b"}
        assert "field3" not in result

    def test_handles_missing_properties_in_schemas(self):
        """Test behavior when schemas lack 'properties' key."""
        data = {"field": "value"}

        # No properties key in schemas
        result = _parse_task_data(data, {}, None)
        assert result == {"field": "value"}
