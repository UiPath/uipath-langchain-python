"""Tests for escalation_tool.py metadata."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationChannelProperties,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    AssetRecipient,
    StandardRecipient,
)

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

        assert result == "resolved@example.com"
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

        assert result == "ResolvedGroup"
        mock_resolve_asset.assert_called_once_with("group_asset", "/Test/Folder")

    @pytest.mark.asyncio
    async def test_resolve_recipient_user_email(self):
        """Test USER_EMAIL type returns value directly."""
        recipient = StandardRecipient(
            type=AgentEscalationRecipientType.USER_EMAIL,
            value="direct@example.com",
        )

        result = await resolve_recipient_value(recipient)

        assert result == "direct@example.com"

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
        tool = await create_escalation_tool(escalation_resource)

        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_tool_type(self, escalation_resource):
        """Test that metadata contains tool_type for span detection."""
        tool = await create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "escalation"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_display_name(self, escalation_resource):
        """Test that metadata contains display_name from app_name."""
        tool = await create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "ApprovalApp"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_channel_type(self, escalation_resource):
        """Test that metadata contains channel_type for span attributes."""
        tool = await create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["channel_type"] == "actionCenter"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_assignee(self, escalation_resource):
        """Test that metadata contains assignee when recipient is USER_EMAIL."""
        tool = await create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["assignee"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_assignee_none_when_no_recipients(
        self, escalation_resource_no_recipient
    ):
        """Test that assignee is None when no recipients configured."""
        tool = await create_escalation_tool(escalation_resource_no_recipient)
        assert tool.metadata is not None
        assert tool.metadata["assignee"] is None
