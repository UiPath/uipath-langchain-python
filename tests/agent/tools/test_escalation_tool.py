"""Tests for escalation_tool.py module."""

import pytest
from unittest.mock import Mock, patch

from uipath.agent.models.agent import (
    AgentEscalationRecipientType,
    StandardRecipient,
    AssetRecipient,
)
from uipath_langchain.agent.tools.escalation_tool import (
    resolve_recipient_value,
    resolve_asset,
)


class TestResolveAsset:
    """Tests for resolve_asset function."""

    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    def test_resolve_asset_success(self, mock_uipath):
        """Test successful asset resolution."""
        # Mock the asset retrieval
        mock_client = Mock()
        mock_result = Mock()
        mock_result.value = "test@example.com"
        mock_client.assets.retrieve.return_value = mock_result
        mock_uipath.return_value = mock_client

        result = resolve_asset("email_asset", "Shared")

        assert result == "test@example.com"
        mock_client.assets.retrieve.assert_called_once_with(
            name="email_asset", folder_path="Shared"
        )

    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    def test_resolve_asset_not_found(self, mock_uipath):
        """Test asset resolution when asset doesn't exist."""
        mock_client = Mock()
        mock_client.assets.retrieve.side_effect = Exception("Asset not found")
        mock_uipath.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            resolve_asset("nonexistent_asset", "Shared")

        assert "Asset not found" in str(exc_info.value)

    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    def test_resolve_asset_raises_error_when_value_is_empty(self, mock_uipath):
        """Test asset resolution raises error when asset value is empty."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.value = None
        mock_client.assets.retrieve.return_value = mock_result
        mock_uipath.return_value = mock_client

        with pytest.raises(ValueError) as exc_info:
            resolve_asset("empty_asset", "Shared")

        assert "has no value" in str(exc_info.value)


class TestResolveRecipientValue:
    """Tests for resolve_recipient_value function."""

    def test_resolve_recipient_value_returns_email_for_user_email_type(self):
        """Test resolving StandardRecipient with USER_EMAIL."""
        recipient = StandardRecipient(
            type=AgentEscalationRecipientType.USER_EMAIL, value="user@example.com"
        )

        result = resolve_recipient_value(recipient)

        assert result == "user@example.com"

    @patch("uipath_langchain.agent.tools.escalation_tool.resolve_asset")
    def test_resolve_recipient_value_calls_resolve_asset_for_asset_recipient(self, mock_resolve_asset):
        """Test resolving AssetRecipient calls resolve_asset."""
        mock_resolve_asset.return_value = "asset@example.com"

        recipient = AssetRecipient(
            type=AgentEscalationRecipientType.ASSET_USER_EMAIL,
            asset_name="email_asset",
            folder_path="Shared",
        )

        result = resolve_recipient_value(recipient)

        assert result == "asset@example.com"
        mock_resolve_asset.assert_called_once_with("email_asset", "Shared")

    @patch("uipath_langchain.agent.tools.escalation_tool.resolve_asset")
    def test_resolve_recipient_value_propagates_error_when_asset_resolution_fails(self, mock_resolve_asset):
        """Test AssetRecipient when asset resolution fails."""
        mock_resolve_asset.side_effect = ValueError("Asset not found")

        recipient = AssetRecipient(
            type=AgentEscalationRecipientType.ASSET_USER_EMAIL,
            asset_name="nonexistent",
            folder_path="Shared",
        )

        with pytest.raises(ValueError) as exc_info:
            resolve_recipient_value(recipient)

        assert "Asset not found" in str(exc_info.value)
