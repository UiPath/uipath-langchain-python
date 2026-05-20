"""Tests for escalation_tool.py metadata."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationChannelProperties,
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    AgentQuickFormChannelProperties,
    AgentQuickFormEscalationChannel,
    ArgumentEmailRecipient,
    ArgumentGroupNameRecipient,
    AssetRecipient,
    CustomAssigneesRecipient,
    RoundRobinRecipient,
    StandardRecipient,
    ToolOutputRecipient,
    WorkloadRecipient,
)
from uipath.platform.action_center.tasks import Task, TaskRecipient, TaskRecipientType

from uipath_langchain.agent.tools.escalation_memory import (
    EscalationMemoryCachedResult,
    _get_user_email,
)
from uipath_langchain.agent.tools.escalation_tool import (
    _build_escalation_memory_payload,
    _build_tool_output_task_recipient,
    _extract_tool_output_value,
    _parse_task_data,
    create_escalation_tool,
    resolve_asset,
    resolve_channel_recipients,
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
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
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
            value="resolved@example.com",
            type=TaskRecipientType.EMAIL,
            displayName="resolved@example.com",
        )
        mock_resolve_asset.assert_called_once_with("email_asset", "/Test/Folder")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
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
            value="ResolvedGroup",
            type=TaskRecipientType.GROUP_NAME,
            displayName="ResolvedGroup",
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
            value="direct@example.com",
            type=TaskRecipientType.EMAIL,
            displayName="direct@example.com",
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

    @pytest.mark.asyncio
    async def test_resolve_recipient_workload(self):
        """WorkloadRecipient resolves to TaskRecipient with WORKLOAD type using displayName."""
        recipient = WorkloadRecipient(
            type=AgentEscalationRecipientType.WORKLOAD,
            value="group-id-1",
            display_name="Support Team",
        )

        result = await resolve_recipient_value(recipient)

        assert result == TaskRecipient(
            value="Support Team",
            type=TaskRecipientType.WORKLOAD,
            displayName="Support Team",
        )

    @pytest.mark.asyncio
    async def test_resolve_recipient_round_robin(self):
        """RoundRobinRecipient resolves to TaskRecipient with ROUND_ROBIN type."""
        recipient = RoundRobinRecipient(
            type=AgentEscalationRecipientType.ROUND_ROBIN,
            value="group-id-1",
            display_name="Support Team",
        )

        result = await resolve_recipient_value(recipient)

        assert result == TaskRecipient(
            value="Support Team",
            type=TaskRecipientType.ROUND_ROBIN,
            displayName="Support Team",
        )

    @pytest.mark.asyncio
    async def test_resolve_recipient_custom_assignees_single(self):
        """A single CustomAssigneesRecipient becomes a one-element Workload assignment."""
        recipient = CustomAssigneesRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            value="alice@example.com",
            displayName="Alice",
        )

        result = await resolve_recipient_value(recipient)

        assert result == TaskRecipient(
            value="alice@example.com",
            values=["alice@example.com"],
            type=TaskRecipientType.WORKLOAD,
            displayName="Alice",
        )

    @pytest.mark.asyncio
    async def test_resolve_recipient_custom_assignees_empty_value_returns_none(self):
        """Empty-value CustomAssignees sentinel resolves to None."""
        recipient = CustomAssigneesRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            value="",
        )

        result = await resolve_recipient_value(recipient)

        assert result is None


class TestResolveChannelRecipients:
    """Tests for the channel-level recipient aggregator."""

    @pytest.mark.asyncio
    async def test_empty_recipients_returns_none(self):
        result = await resolve_channel_recipients([])
        assert result is None

    @pytest.mark.asyncio
    async def test_single_workload_recipient_delegates_to_resolve_recipient_value(self):
        recipient = WorkloadRecipient(
            type=AgentEscalationRecipientType.WORKLOAD,
            value="group-1",
            display_name="Support Team",
        )

        result = await resolve_channel_recipients([recipient])

        assert result == TaskRecipient(
            value="Support Team",
            type=TaskRecipientType.WORKLOAD,
            displayName="Support Team",
        )

    @pytest.mark.asyncio
    async def test_multiple_custom_assignees_collected_into_single_workload(self):
        """Multiple CustomAssignees recipients aggregate into one Workload + values list."""
        recipients: list[AgentEscalationRecipient] = [
            CustomAssigneesRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                value="alice@example.com",
                displayName="Alice",
            ),
            CustomAssigneesRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                value="bob@example.com",
                displayName="Bob",
            ),
        ]

        result = await resolve_channel_recipients(recipients)

        assert result == TaskRecipient(
            value="alice@example.com",
            values=["alice@example.com", "bob@example.com"],
            type=TaskRecipientType.WORKLOAD,
        )

    @pytest.mark.asyncio
    async def test_single_custom_assignee_collected_into_single_workload(self):
        recipients: list[AgentEscalationRecipient] = [
            CustomAssigneesRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                value="alice@example.com",
            ),
        ]

        result = await resolve_channel_recipients(recipients)

        assert result == TaskRecipient(
            value="alice@example.com",
            values=["alice@example.com"],
            type=TaskRecipientType.WORKLOAD,
        )

    @pytest.mark.asyncio
    async def test_custom_assignees_filters_empty_sentinel_values(self):
        """Empty-value sentinel CustomAssignees are skipped, not included in the array."""
        recipients: list[AgentEscalationRecipient] = [
            CustomAssigneesRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                value="",
            ),
            CustomAssigneesRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                value="alice@example.com",
            ),
        ]

        result = await resolve_channel_recipients(recipients)

        assert result == TaskRecipient(
            value="alice@example.com",
            values=["alice@example.com"],
            type=TaskRecipientType.WORKLOAD,
        )

    @pytest.mark.asyncio
    async def test_all_empty_custom_assignees_returns_none(self):
        recipients: list[AgentEscalationRecipient] = [
            CustomAssigneesRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                value="",
            ),
        ]

        result = await resolve_channel_recipients(recipients)
        assert result is None

    @pytest.mark.asyncio
    async def test_standard_recipient_uses_first_recipient_only(self):
        """Non-CustomAssignees channels only use recipients[0]."""
        recipient = StandardRecipient(
            type=AgentEscalationRecipientType.USER_EMAIL,
            value="alice@example.com",
        )

        result = await resolve_channel_recipients([recipient])

        assert result == TaskRecipient(
            value="alice@example.com",
            type=TaskRecipientType.EMAIL,
            displayName="alice@example.com",
        )


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

    @pytest.fixture
    def quick_form_escalation_resource(self):
        """Create a quick-form escalation resource (channel has no app_name)."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentQuickFormEscalationChannel(
                    name="Escalation",
                    type="actionCenterQuickForm",
                    description="Quick Form channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentQuickFormChannelProperties(
                        form_schema={"schemaId": "schema-123", "fields": []},
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
    async def test_escalation_tool_metadata_display_name_falls_back_to_channel_name(
        self, quick_form_escalation_resource
    ):
        """Quick-form channels have no app_name; display_name uses the channel name."""
        tool = create_escalation_tool(quick_form_escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "Escalation"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_channel_type(self, escalation_resource):
        """Test that metadata contains channel_type for span attributes."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["channel_type"] == "actionCenter"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_span_context(self, escalation_resource):
        """Test that metadata contains a span context carrier for memory ingest."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert "_span_context" in tool.metadata
        assert isinstance(tool.metadata["_span_context"], dict)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] == TaskRecipient(
            value="user@example.com",
            type=TaskRecipientType.EMAIL,
            displayName="user@example.com",
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource_no_recipient)

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
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
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
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
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
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
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert isinstance(result, dict)
        assert result["outcome"] == "approve"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert mock_interrupt.called

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_raises_when_task_is_deleted(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that escalation tool raises AgentRuntimeError when task is deleted."""
        from uipath_langchain.agent.exceptions import AgentRuntimeError

        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = True
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_dict_result_without_is_deleted_defaults_to_false(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that a dict result without is_deleted is accepted and defaults to False."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        # Return a plain dict without is_deleted — exercises the TypeAdapter path
        mock_interrupt.return_value = {
            "action": "approve",
            "data": {"approved": True, "reason": "looks good"},
        }

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["outcome"] == "approve"
        assert result["output"] == {"approved": True, "reason": "looks good"}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_with_outcome_mapping_end(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool with outcome mapping that ends agent."""
        from uipath_langchain.agent.exceptions import AgentRuntimeError

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

        # Invoke through the wrapper - should raise AgentRuntimeError
        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert mock_interrupt.called

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    async def test_cached_escalation_uses_outcome_mapping(
        self, mock_check_memory_cache: AsyncMock
    ):
        """Test cached outcomes follow the same outcome mapping as live results."""
        from uipath_langchain.agent.exceptions import AgentRuntimeError

        mock_check_memory_cache.return_value = EscalationMemoryCachedResult(
            output={"approved": True},
            outcome="approve",
        )

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
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.get_execution_folder_path")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    async def test_cache_lookup_uses_memory_folder_path(
        self,
        mock_check_memory_cache: AsyncMock,
        mock_get_execution_folder_path: MagicMock,
    ):
        """Test escalation memory calls use the memory folder, not task folder."""
        mock_get_execution_folder_path.return_value = "/Execution/Folder"
        mock_check_memory_cache.return_value = EscalationMemoryCachedResult(
            output={"approved": True},
            outcome="approve",
        )

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
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
            properties={
                "memory": {
                    "isEnabled": True,
                    "memorySpaceId": "space-123",
                    "memorySpaceName": "MemorySpace",
                    "folderPath": "/Memory/Folder",
                }
            },
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result == {
            "output": {"approved": True},
            "outcome": "approve",
            "task_id": None,
            "assigned_to": None,
        }
        mock_check_memory_cache.assert_awaited_once()
        assert mock_check_memory_cache.await_args is not None
        assert (
            mock_check_memory_cache.await_args.kwargs["folder_path"] == "/Memory/Folder"
        )
        assert (
            mock_check_memory_cache.await_args.kwargs["memory_space_name"]
            == "MemorySpace"
        )


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
    async def test_wrapper_returns_task_id_and_assigned_to(self, escalation_resource):
        """Test that wrapper result includes task_id and assigned_to from Task."""
        tool = create_escalation_tool(escalation_resource)

        # Mock ainvoke on the class to test the wrapper in isolation
        mock_ainvoke = AsyncMock(
            return_value={
                "action": "continue",
                "output": {"reason": "looks good"},
                "outcome": "approve",
                "task_id": 12345,
                "assigned_to": "user@example.com",
            }
        )

        call = ToolCall(args={}, id="test-call", name=tool.name)
        with patch.object(type(tool), "ainvoke", mock_ainvoke):
            result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["task_id"] == 12345
        assert result["assigned_to"] == "user@example.com"
        assert result["outcome"] == "approve"

    @pytest.mark.asyncio
    async def test_wrapper_handles_missing_assigned_to_user(self, escalation_resource):
        """Test that wrapper handles None assigned_to_user gracefully."""
        tool = create_escalation_tool(escalation_resource)

        # Mock ainvoke on the class to test the wrapper in isolation
        mock_ainvoke = AsyncMock(
            return_value={
                "action": "continue",
                "output": {},
                "outcome": "reject",
                "task_id": 99999,
                "assigned_to": None,
            }
        )

        call = ToolCall(args={}, id="test-call", name=tool.name)
        with patch.object(type(tool), "ainvoke", mock_ainvoke):
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
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
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
        mock_result.is_deleted = False
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
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_creates_task_with_execution_folder_path(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that tasks.create_async receives app_folder_path from the execution environment."""
        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 555
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.assigned_to_user = None
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        create_call_kwargs = mock_client.tasks.create_async.call_args[1]
        assert create_call_kwargs["app_folder_path"] == "/Test/Folder"

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

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool.get_current_span_and_trace_ids"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool._ingest_escalation_memory")
    @patch("uipath_langchain.agent.tools.escalation_tool._resolve_user_id")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_memory_ingest_uses_traced_escalation_span_context(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_check_memory_cache,
        mock_resolve_user_id,
        mock_ingest_memory,
        mock_get_current_span_and_trace_ids,
    ):
        """Escalation memory ingest should use the escalationTool child span."""
        mock_check_memory_cache.return_value = None
        mock_resolve_user_id.return_value = "cef1337c-3456-4ae9-81c9-30d033dc2bef"
        mock_ingest_memory.return_value = None
        mock_get_current_span_and_trace_ids.return_value = ("wrong-span", "wrong-trace")

        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.completed_by_user = {"emailAddress": "reviewer@example.com"}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        resource = AgentEscalationResourceConfig(
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
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        assert tool.metadata is not None
        tool.metadata["_span_context"]["parent_span_id"] = "3a064d559eca5d62"
        tool.metadata["_span_context"]["trace_id"] = "5d3feebba60343dfb9364b89ee304a5b"

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_get_current_span_and_trace_ids.assert_not_called()
        mock_ingest_memory.assert_awaited_once()
        assert mock_ingest_memory.await_args is not None
        assert (
            mock_ingest_memory.await_args.kwargs["parent_span_id"] == "3a064d559eca5d62"
        )
        assert (
            mock_ingest_memory.await_args.kwargs["trace_id"]
            == "5d3feebba60343dfb9364b89ee304a5b"
        )
        assert tool.metadata["_span_context"] == {}

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool.get_current_span_and_trace_ids"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool._ingest_escalation_memory")
    @patch("uipath_langchain.agent.tools.escalation_tool._resolve_user_id")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_memory_ingest_falls_back_to_current_span_context(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_check_memory_cache,
        mock_resolve_user_id,
        mock_ingest_memory,
        mock_get_current_span_and_trace_ids,
    ):
        """Escalation memory ingest should fall back when metadata is incomplete."""
        mock_check_memory_cache.return_value = None
        mock_resolve_user_id.return_value = None
        mock_ingest_memory.return_value = None
        mock_get_current_span_and_trace_ids.return_value = (
            "fallback-span",
            "fallback-trace",
        )

        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.completed_by_user = {"displayName": "Reviewer"}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        resource = AgentEscalationResourceConfig(
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
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_get_current_span_and_trace_ids.assert_called_once()
        mock_ingest_memory.assert_awaited_once()
        assert mock_ingest_memory.await_args is not None
        assert mock_ingest_memory.await_args.kwargs["parent_span_id"] == "fallback-span"
        assert mock_ingest_memory.await_args.kwargs["trace_id"] == "fallback-trace"
        assert mock_ingest_memory.await_args.kwargs["user_id"] is None

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool.get_current_span_and_trace_ids"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool._ingest_escalation_memory")
    @patch("uipath_langchain.agent.tools.escalation_tool._resolve_user_id")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_memory_ingest_skips_when_span_context_is_unavailable(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_check_memory_cache,
        mock_resolve_user_id,
        mock_ingest_memory,
        mock_get_current_span_and_trace_ids,
    ):
        """Escalation memory ingest should be skipped without trace provenance."""
        mock_check_memory_cache.return_value = None
        mock_resolve_user_id.return_value = None
        mock_get_current_span_and_trace_ids.return_value = (None, None)

        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.completed_by_user = {"displayName": "Reviewer"}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        resource = AgentEscalationResourceConfig(
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
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["output"] == {}
        assert result["outcome"] == "approve"
        mock_get_current_span_and_trace_ids.assert_called_once()
        mock_ingest_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wrapper_requires_metadata(self, escalation_resource):
        tool = create_escalation_tool(escalation_resource)
        tool.metadata = None
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(
            RuntimeError,
            match="Tool metadata is required for task_title resolution",
        ):
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


class TestEscalationMemoryPayload:
    """Test escalation memory ingest payload shape."""

    def test_builds_trace_and_search_payloads(self):
        """Test memory ingest matches the escalation memory service contract."""
        serialized_input = {
            "request_details": "User requested escalation before answering."
        }
        escalation_output = {"reviewer_comment": "approve"}

        answer, attributes = _build_escalation_memory_payload(
            serialized_input,
            escalation_output,
            "Approve",
        )

        assert answer == {
            "output": {"reviewer_comment": "approve"},
            "outcome": "Approve",
        }
        assert attributes == {"arguments": serialized_input}
        assert "escalation-input" not in attributes


class TestQuickFormEscalation:
    """QuickForm channel (actionCenterQuickForm) path through create_escalation_tool."""

    @pytest.fixture
    def quick_form_schema(self):
        return {
            "schemaId": "00000000-0000-0000-0000-000000000abc",
            "fields": [{"name": "decision", "type": "string"}],
            "outcomes": ["approve", "reject"],
        }

    @pytest.fixture
    def quick_form_channel_dict(self, quick_form_schema):
        return {
            "name": "quick_form_channel",
            "type": "actionCenterQuickForm",
            "description": "Quick-form channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "schema": quick_form_schema,
                "isActionableMessageEnabled": False,
                "actionableMessageMetaData": None,
            },
            "recipients": [],
        }

    @pytest.fixture
    def quick_form_resource(self, quick_form_channel_dict):
        return AgentEscalationResourceConfig(
            name="quick_form_approval",
            description="Request quick-form approval",
            channels=[AgentQuickFormEscalationChannel(**quick_form_channel_dict)],
        )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_dispatches_to_create_quickform_async(
        self,
        mock_interrupt,
        mock_uipath_class,
        quick_form_resource,
        quick_form_schema,
    ):
        task = _make_mock_task(id=777, key="task-key-777")
        mock_client = MagicMock()
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(quick_form_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_client.tasks.create_quickform_async.assert_called_once()
        mock_client.tasks.create_async.assert_not_called()

        kwargs = mock_client.tasks.create_quickform_async.call_args[1]
        assert kwargs["task_schema_key"] == "00000000-0000-0000-0000-000000000abc"
        assert kwargs["schema"] == quick_form_schema
        assert kwargs["folder_path"] == "/Test/Folder"
        assert "app_name" not in kwargs
        assert "app_folder_path" not in kwargs

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_wait_escalation_app_name_is_none_for_quick_form(
        self, mock_interrupt, mock_uipath_class, quick_form_resource
    ):
        from uipath.platform.common import WaitEscalation

        task = _make_mock_task(id=778)
        mock_client = MagicMock()
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(quick_form_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_interrupt.assert_called_once()
        interrupt_arg = mock_interrupt.call_args[0][0]
        assert isinstance(interrupt_arg, WaitEscalation)
        assert interrupt_arg.app_name is None
        assert interrupt_arg.action == task

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_outcome_mapping_end_terminates_agent(
        self,
        mock_interrupt,
        mock_uipath_class,
        quick_form_channel_dict,
    ):
        from uipath_langchain.agent.exceptions import AgentRuntimeError

        channel = dict(quick_form_channel_dict)
        channel["outcomeMapping"] = {"approve": "end", "reject": "continue"}
        resource = AgentEscalationResourceConfig(
            name="quick_form_approval",
            description="Request quick-form approval",
            channels=[AgentQuickFormEscalationChannel(**channel)],
        )

        task = _make_mock_task(id=779)
        mock_client = MagicMock()
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_tool_metadata_for_quick_form_resource(self, quick_form_resource):
        tool = create_escalation_tool(quick_form_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "escalation"
        assert tool.metadata["channel_type"] == "actionCenterQuickForm"
        assert "_span_context" in tool.metadata
        assert "_bts_context" in tool.metadata

    async def test_missing_schema_id_raises_on_construction(
        self, quick_form_channel_dict
    ):
        from uipath_langchain.agent.exceptions import AgentStartupError

        channel = dict(quick_form_channel_dict)
        channel["properties"] = {
            "schema": {"fields": [], "outcomes": []},
            "isActionableMessageEnabled": False,
            "actionableMessageMetaData": None,
        }
        resource = AgentEscalationResourceConfig(
            name="quick_form_approval",
            description="Request quick-form approval",
            channels=[AgentQuickFormEscalationChannel(**channel)],
        )

        with pytest.raises(AgentStartupError) as exc_info:
            create_escalation_tool(resource)

        assert "INVALID_TOOL_CONFIG" in exc_info.value.error_info.code

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_action_center_channel_does_not_dispatch_to_quickform(
        self, mock_interrupt, mock_uipath_class
    ):
        resource = AgentEscalationResourceConfig(
            name="action_center_approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center_channel",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp", app_version=1
                    ),
                    recipients=[],
                )
            ],
        )

        task = _make_mock_task(id=780)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_client.tasks.create_async.assert_called_once()
        mock_client.tasks.create_quickform_async.assert_not_called()


class TestExtractToolOutputValue:
    """Tests for the helper that walks message history to extract tool output values."""

    def test_extracts_top_level_field_from_json_content(self):
        messages = [
            HumanMessage(content="hello"),
            ToolMessage(
                name="API workflow A",
                tool_call_id="1",
                content='{"includeEmails": ["alice@x.com", "bob@x.com"], "expiresAt": "2026-01-01"}',
            ),
        ]
        result = _extract_tool_output_value(messages, "API workflow A", "includeEmails")
        assert result == ["alice@x.com", "bob@x.com"]

    def test_returns_whole_parsed_content_when_path_is_empty(self):
        messages = [
            ToolMessage(
                name="getApprovers",
                tool_call_id="1",
                content='["alice@x.com", "bob@x.com"]',
            ),
        ]
        result = _extract_tool_output_value(messages, "getApprovers", "")
        assert result == ["alice@x.com", "bob@x.com"]

    def test_picks_latest_invocation_when_called_multiple_times(self):
        messages = [
            ToolMessage(
                name="A", tool_call_id="1", content='{"emails": ["old@x.com"]}'
            ),
            ToolMessage(
                name="A", tool_call_id="2", content='{"emails": ["new@x.com"]}'
            ),
        ]
        result = _extract_tool_output_value(messages, "A", "emails")
        assert result == ["new@x.com"]

    def test_raises_when_tool_was_never_called(self):
        messages = [
            ToolMessage(name="getUsers", tool_call_id="1", content="[]"),
        ]
        with pytest.raises(ValueError, match="has not been called yet"):
            _extract_tool_output_value(messages, "API workflow A", "emails")

    def test_raises_when_field_missing_from_output(self):
        messages = [
            ToolMessage(name="A", tool_call_id="1", content='{"otherField": []}'),
        ]
        with pytest.raises(ValueError, match="does not contain field 'emails'"):
            _extract_tool_output_value(messages, "A", "emails")

    def test_raises_when_output_is_not_a_json_object_but_path_requested(self):
        messages = [
            ToolMessage(name="A", tool_call_id="1", content='["a", "b"]'),
        ]
        with pytest.raises(ValueError, match="is not a JSON object"):
            _extract_tool_output_value(messages, "A", "emails")

    def test_handles_non_json_string_content_gracefully(self):
        # If the content isn't JSON, treat it as a raw string (the whole "output").
        messages = [
            ToolMessage(name="A", tool_call_id="1", content="some raw text"),
        ]
        # With no path, the raw string is returned.
        result = _extract_tool_output_value(messages, "A", "")
        assert result == "some raw text"

    def test_ignores_non_matching_tool_messages(self):
        messages = [
            ToolMessage(
                name="otherTool", tool_call_id="1", content='{"emails": ["wrong"]}'
            ),
            ToolMessage(
                name="A", tool_call_id="2", content='{"emails": ["right@x.com"]}'
            ),
            AIMessage(content="thinking"),
        ]
        result = _extract_tool_output_value(messages, "A", "emails")
        assert result == ["right@x.com"]

    def test_returns_empty_string_when_content_is_empty_and_no_path(self):
        # Empty content with no path falls through the JSON parse to the raw "" string.
        messages = [ToolMessage(name="A", tool_call_id="1", content="")]
        result = _extract_tool_output_value(messages, "A", "")
        assert result == ""

    def test_raises_when_malformed_json_and_path_requested(self):
        # Malformed JSON content falls back to the raw string; attempting to extract a
        # path then fails the same way as any other non-object output.
        messages = [
            ToolMessage(name="A", tool_call_id="1", content='{"emails": ["a@x.com"'),
        ]
        with pytest.raises(ValueError, match="is not a JSON object"):
            _extract_tool_output_value(messages, "A", "emails")

    def test_returns_null_when_field_value_is_explicitly_null(self):
        # A null field value at the requested path is returned as Python None; the
        # downstream recipient builder is the one that rejects None/empty values.
        messages = [
            ToolMessage(name="A", tool_call_id="1", content='{"emails": null}'),
        ]
        result = _extract_tool_output_value(messages, "A", "emails")
        assert result is None

    def test_uses_non_string_content_directly_without_parsing(self):
        # When ToolMessage.content is already structured (e.g. dict), use it
        # as-is rather than attempting json.loads on a non-string.
        messages = [
            ToolMessage(
                name="A",
                tool_call_id="1",
                content=[{"type": "text", "text": '{"emails": ["a@x.com"]}'}],
            ),
        ]
        # Non-string content is returned verbatim when no path is requested.
        result = _extract_tool_output_value(messages, "A", "")
        assert result == [{"type": "text", "text": '{"emails": ["a@x.com"]}'}]


class TestBuildToolOutputTaskRecipient:
    """Tests for _build_tool_output_task_recipient single-value & edge branches."""

    def test_returns_group_id_recipient_for_group_id_criteria(self):
        # Single string mapped through the GROUP_ID branch.
        result = _build_tool_output_task_recipient(
            AgentEscalationRecipientType.GROUP_ID, "group-123"
        )
        assert result is not None
        assert result.value == "group-123"
        assert result.type == TaskRecipientType.GROUP_ID

    def test_returns_user_id_recipient_for_user_id_criteria(self):
        result = _build_tool_output_task_recipient(
            AgentEscalationRecipientType.USER_ID, "user-1"
        )
        assert result is not None
        assert result.value == "user-1"
        assert result.type == TaskRecipientType.USER_ID

    def test_workload_string_single_value(self):
        result = _build_tool_output_task_recipient(
            AgentEscalationRecipientType.WORKLOAD, "workload-1"
        )
        assert result is not None
        assert result.values == ["workload-1"]
        assert result.type == TaskRecipientType.WORKLOAD

    def test_round_robin_string_single_value(self):
        result = _build_tool_output_task_recipient(
            AgentEscalationRecipientType.ROUND_ROBIN, "rr-1"
        )
        assert result is not None
        assert result.type == TaskRecipientType.ROUND_ROBIN

    def test_custom_assignees_splits_comma_separated_string(self):
        result = _build_tool_output_task_recipient(
            AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            "alice@example.com, bob@example.com , ,",
        )
        assert result is not None
        assert result.values == ["alice@example.com", "bob@example.com"]
        assert result.type == TaskRecipientType.WORKLOAD

    def test_custom_assignees_returns_none_for_only_separators(self):
        # A string with only commas/whitespace yields no valid emails -> None.
        result = _build_tool_output_task_recipient(
            AgentEscalationRecipientType.CUSTOM_ASSIGNEES, ", , ,"
        )
        assert result is None

    def test_returns_none_for_unknown_criteria_type(self):
        # Recipient types that aren't single-valued criteria (e.g. an asset
        # type accidentally routed through here) fall through to a None.
        result = _build_tool_output_task_recipient(
            AgentEscalationRecipientType.ASSET_USER_EMAIL, "value"
        )
        assert result is None

    def test_raises_when_resolved_string_value_is_empty(self):
        with pytest.raises(ValueError, match="empty value"):
            _build_tool_output_task_recipient(AgentEscalationRecipientType.USER_ID, "")

    def test_raises_when_resolved_value_is_none_string_coerce(self):
        with pytest.raises(ValueError, match="empty value"):
            _build_tool_output_task_recipient(
                AgentEscalationRecipientType.WORKLOAD, None
            )


class TestArgumentRecipientResolutionMissing:
    """Argument-name recipients raise loudly when the named input is missing."""

    @pytest.mark.asyncio
    async def test_argument_email_raises_when_input_missing(self):
        recipient = ArgumentEmailRecipient(
            type=AgentEscalationRecipientType.ARGUMENT_EMAIL,
            argument_path="user.email",
        )
        with pytest.raises(ValueError, match="no value in agent input"):
            await resolve_recipient_value(recipient, input_args={})

    @pytest.mark.asyncio
    async def test_argument_group_name_raises_when_input_missing(self):
        recipient = ArgumentGroupNameRecipient(
            type=AgentEscalationRecipientType.ARGUMENT_GROUP_NAME,
            argument_path="group.name",
        )
        with pytest.raises(ValueError, match="no value in agent input"):
            await resolve_recipient_value(recipient, input_args={})

    @pytest.mark.asyncio
    async def test_argument_email_resolves_value_from_input(self):
        recipient = ArgumentEmailRecipient(
            type=AgentEscalationRecipientType.ARGUMENT_EMAIL,
            argument_path="approver.email",
        )
        result = await resolve_recipient_value(
            recipient,
            input_args={"approver": {"email": "boss@example.com"}},
        )
        assert result is not None
        assert result.value == "boss@example.com"
        assert result.type == TaskRecipientType.EMAIL

    @pytest.mark.asyncio
    async def test_argument_group_name_resolves_value_from_input(self):
        recipient = ArgumentGroupNameRecipient(
            type=AgentEscalationRecipientType.ARGUMENT_GROUP_NAME,
            argument_path="dept.team",
        )
        result = await resolve_recipient_value(
            recipient,
            input_args={"dept": {"team": "Finance"}},
        )
        assert result is not None
        assert result.value == "Finance"
        assert result.type == TaskRecipientType.GROUP_NAME


class TestToolOutputRecipientResolution:
    """Tests for resolving ToolOutputRecipient via resolve_recipient_value."""

    @pytest.mark.asyncio
    async def test_custom_assignees_with_array_of_emails(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            source="toolOutput",
            tool_name="API workflow A",
            output_path="includeEmails",
        )
        messages = [
            ToolMessage(
                name="API workflow A",
                tool_call_id="1",
                content='{"includeEmails": ["alice@x.com", "bob@x.com"]}',
            ),
        ]

        result = await resolve_recipient_value(recipient, tool_messages=messages)

        assert result == TaskRecipient(
            value="alice@x.com",
            values=["alice@x.com", "bob@x.com"],
            type=TaskRecipientType.WORKLOAD,
        )

    @pytest.mark.asyncio
    async def test_custom_assignees_with_comma_separated_string(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            source="toolOutput",
            tool_name="A",
            output_path="emails",
        )
        messages = [
            ToolMessage(
                name="A",
                tool_call_id="1",
                content='{"emails": "alice@x.com, bob@x.com,  carol@x.com"}',
            ),
        ]

        result = await resolve_recipient_value(recipient, tool_messages=messages)

        assert result is not None
        assert result.type == TaskRecipientType.WORKLOAD
        assert result.values == ["alice@x.com", "bob@x.com", "carol@x.com"]

    @pytest.mark.asyncio
    async def test_workload_with_single_group_name_string(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.WORKLOAD,
            source="toolOutput",
            tool_name="findGroup",
            output_path="groupName",
        )
        messages = [
            ToolMessage(
                name="findGroup",
                tool_call_id="1",
                content='{"groupName": "Support Team"}',
            ),
        ]

        result = await resolve_recipient_value(recipient, tool_messages=messages)

        assert result == TaskRecipient(
            value="Support Team",
            values=["Support Team"],
            type=TaskRecipientType.WORKLOAD,
        )

    @pytest.mark.asyncio
    async def test_round_robin_with_single_group_name(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.ROUND_ROBIN,
            source="toolOutput",
            tool_name="findGroup",
            output_path="groupName",
        )
        messages = [
            ToolMessage(
                name="findGroup",
                tool_call_id="1",
                content='{"groupName": "Support Team"}',
            ),
        ]

        result = await resolve_recipient_value(recipient, tool_messages=messages)

        assert result == TaskRecipient(
            value="Support Team",
            values=["Support Team"],
            type=TaskRecipientType.ROUND_ROBIN,
        )

    @pytest.mark.asyncio
    async def test_user_with_string_value(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.USER_ID,
            source="toolOutput",
            tool_name="findUser",
            output_path="userId",
        )
        messages = [
            ToolMessage(
                name="findUser",
                tool_call_id="1",
                content='{"userId": "user-123"}',
            ),
        ]

        result = await resolve_recipient_value(recipient, tool_messages=messages)

        assert result == TaskRecipient(value="user-123", type=TaskRecipientType.USER_ID)

    @pytest.mark.asyncio
    async def test_raises_when_tool_not_called(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            source="toolOutput",
            tool_name="API workflow A",
            output_path="includeEmails",
        )
        with pytest.raises(ValueError, match="has not been called yet"):
            await resolve_recipient_value(recipient, tool_messages=[])

    @pytest.mark.asyncio
    async def test_raises_when_resolved_value_is_empty_list(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            source="toolOutput",
            tool_name="A",
            output_path="emails",
        )
        messages = [
            ToolMessage(name="A", tool_call_id="1", content='{"emails": []}'),
        ]
        with pytest.raises(ValueError, match="empty list"):
            await resolve_recipient_value(recipient, tool_messages=messages)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "criteria",
        [
            AgentEscalationRecipientType.USER_ID,
            AgentEscalationRecipientType.GROUP_ID,
            AgentEscalationRecipientType.ROUND_ROBIN,
        ],
    )
    async def test_raises_when_list_resolved_for_single_valued_criteria(self, criteria):
        # Single-valued criteria (User/Group/RoundRobin) cannot accept a list of
        # recipients; silently demoting to WORKLOAD would change the assignment
        # semantics. The resolver should raise instead.
        recipient = ToolOutputRecipient(
            type=criteria,
            source="toolOutput",
            tool_name="A",
            output_path="ids",
        )
        messages = [
            ToolMessage(
                name="A",
                tool_call_id="1",
                content='{"ids": ["one", "two"]}',
            ),
        ]
        with pytest.raises(ValueError, match="expects a single value"):
            await resolve_recipient_value(recipient, tool_messages=messages)

    @pytest.mark.asyncio
    async def test_tool_name_matches_sanitized_form(self):
        # ToolMessage.name carries the sanitized tool name (matches the form
        # used at tool-call time). Recipient configs may carry either the
        # display name or the sanitized form — both must resolve.
        from uipath_langchain.agent.tools.utils import sanitize_tool_name

        display_name = "Get Users Info"
        sanitized = sanitize_tool_name(display_name)
        assert sanitized != display_name  # sanity check: forms differ

        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.USER_ID,
            source="toolOutput",
            tool_name=display_name,
            output_path="userId",
        )
        messages = [
            ToolMessage(name=sanitized, tool_call_id="1", content='{"userId": "u-1"}'),
        ]

        result = await resolve_recipient_value(recipient, tool_messages=messages)
        assert result is not None
        assert result.value == "u-1"
        assert result.type == TaskRecipientType.USER_ID


class TestResolveChannelRecipientsWithToolOutput:
    """Tests for resolve_channel_recipients with tool-output bindings."""

    @pytest.mark.asyncio
    async def test_tool_output_recipient_delegates_to_resolver(self):
        recipient = ToolOutputRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            source="toolOutput",
            tool_name="A",
            output_path="emails",
        )
        messages = [
            ToolMessage(
                name="A",
                tool_call_id="1",
                content='{"emails": ["a@b.com", "c@d.com"]}',
            ),
        ]

        result = await resolve_channel_recipients([recipient], tool_messages=messages)

        assert result is not None
        assert result.values == ["a@b.com", "c@d.com"]

    @pytest.mark.asyncio
    async def test_tool_output_takes_precedence_over_custom_assignees_aggregation(self):
        # Even if subsequent recipients are CustomAssigneesRecipient, a leading
        # ToolOutputRecipient owns the entire channel resolution.
        recipients: list[AgentEscalationRecipient] = [
            ToolOutputRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                source="toolOutput",
                tool_name="A",
                output_path="emails",
            ),
            CustomAssigneesRecipient(
                type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                value="manual@x.com",
            ),
        ]
        messages = [
            ToolMessage(
                name="A",
                tool_call_id="1",
                content='{"emails": ["resolved@x.com"]}',
            ),
        ]

        result = await resolve_channel_recipients(recipients, tool_messages=messages)

        assert result is not None
        # Tool-output wins; manual recipient is ignored.
        assert result.values == ["resolved@x.com"]


class TestEscalationToolDescriptionAugmentation:
    """Tests for the LLM-facing description hint added when tool-output bindings exist."""

    def _make_channel(
        self, recipients: list[AgentEscalationRecipient]
    ) -> AgentEscalationChannel:
        return AgentEscalationChannel(
            id="ch-1",
            type="actionCenter",
            name="ch",
            description="",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            recipients=recipients,
            properties=AgentEscalationChannelProperties(
                app_name="my-app",
                resource_key="rk",
                folder_name=None,
                app_version=1,
                is_actionable_message_enabled=False,
                actionable_message_meta_data=None,
            ),
        )

    def _make_resource(
        self, channel: AgentEscalationChannel
    ) -> AgentEscalationResourceConfig:
        return AgentEscalationResourceConfig(
            resourceType="escalation",
            id="esc-1",
            name="approve_expense",
            description="Escalate an expense for approval.",
            channels=[channel],
        )

    def test_description_unchanged_when_no_tool_output_bindings(self):
        channel = self._make_channel(
            recipients=[
                StandardRecipient(
                    type=AgentEscalationRecipientType.USER_ID,
                    value="u1",
                    displayName="User 1",
                )
            ]
        )
        tool = create_escalation_tool(self._make_resource(channel))
        assert tool.description == "Escalate an expense for approval."
        assert "Recipient routing notes" not in tool.description

    def test_description_includes_dependency_hint_for_tool_output_binding(self):
        channel = self._make_channel(
            recipients=[
                ToolOutputRecipient(
                    type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                    source="toolOutput",
                    tool_name="API workflow A",
                    output_path="includeEmails",
                )
            ]
        )
        tool = create_escalation_tool(self._make_resource(channel))
        assert "Recipient routing notes" in tool.description
        assert "API workflow A" in tool.description
        assert "includeEmails" in tool.description

    def test_description_deduplicates_repeated_tool_dependencies(self):
        channel = self._make_channel(
            recipients=[
                ToolOutputRecipient(
                    type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                    source="toolOutput",
                    tool_name="A",
                    output_path="emails",
                ),
                ToolOutputRecipient(
                    type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
                    source="toolOutput",
                    tool_name="A",
                    output_path="emails",
                ),
            ]
        )
        tool = create_escalation_tool(self._make_resource(channel))
        # The hint mentions tool A exactly once.
        assert tool.description.count("`A`") == 1
