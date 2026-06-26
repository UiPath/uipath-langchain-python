"""Tests for escalation_recipient.py recipient resolution."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentEscalationRecipientType,
    ArgumentEmailRecipient,
    ArgumentGroupNameRecipient,
    AssetRecipient,
    CustomAssigneesRecipient,
    RoundRobinRecipient,
    StandardRecipient,
    WorkloadRecipient,
)
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType

from uipath_langchain.agent.tools.escalation_memory import _get_user_email
from uipath_langchain.agent.tools.escalation_recipient import (
    _build_llm_recipient,
    _resolve_asset,
    resolve_recipient_value,
)


class TestResolveAsset:
    """Test the resolve_asset function."""

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_recipient.UiPath")
    async def test_resolve_asset_success(self, mock_uipath_class):
        """Test successful asset retrieval."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_result = MagicMock()
        mock_result.value = "test@example.com"
        mock_client.assets.retrieve_async = AsyncMock(return_value=mock_result)

        result = await _resolve_asset("email_asset", "/Test/Folder")

        assert result == "test@example.com"
        mock_client.assets.retrieve_async.assert_called_once_with(
            name="email_asset", folder_path="/Test/Folder"
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_recipient.UiPath")
    async def test_resolve_asset_no_value(self, mock_uipath_class):
        """Test asset with no value raises ValueError."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_result = MagicMock()
        mock_result.value = None
        mock_client.assets.retrieve_async = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError) as exc_info:
            await _resolve_asset("empty_asset", "/Test/Folder")

        assert "Asset 'empty_asset' has no value configured" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_recipient.UiPath")
    async def test_resolve_asset_not_found(self, mock_uipath_class):
        """Test asset not found raises ValueError."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.assets.retrieve_async = AsyncMock(return_value=None)

        with pytest.raises(ValueError) as exc_info:
            await _resolve_asset("missing_asset", "/Test/Folder")

        assert "Asset 'missing_asset' has no value configured" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_recipient.UiPath")
    async def test_resolve_asset_retrieval_exception(self, mock_uipath_class):
        """Test exception during asset retrieval raises ValueError with context."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.assets.retrieve_async = AsyncMock(
            side_effect=Exception("Connection error")
        )

        with pytest.raises(ValueError) as exc_info:
            await _resolve_asset("problem_asset", "/Test/Folder")

        assert (
            "Failed to resolve asset 'problem_asset' in folder '/Test/Folder'"
            in str(exc_info.value)
        )
        assert "Connection error" in str(exc_info.value)


class TestResolveRecipientValue:
    """Test the resolve_recipient_value function."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
    @patch("uipath_langchain.agent.tools.escalation_recipient._resolve_asset")
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
    @patch("uipath_langchain.agent.tools.escalation_recipient._resolve_asset")
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
    @patch("uipath_langchain.agent.tools.escalation_recipient._resolve_asset")
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
    async def test_resolve_recipient_custom_assignees_returns_none(self):
        """A CustomAssigneesRecipient resolves to None at the design-time resolver."""
        recipient = CustomAssigneesRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            value="alice@example.com",
            displayName="Alice",
        )

        result = await resolve_recipient_value(recipient)

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_recipient_custom_assignees_empty_value_returns_none(self):
        """Empty-value CustomAssignees sentinel resolves to None."""
        recipient = CustomAssigneesRecipient(
            type=AgentEscalationRecipientType.CUSTOM_ASSIGNEES,
            value="",
        )

        result = await resolve_recipient_value(recipient)

        assert result is None


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


class TestBuildLlmRecipient:
    """Tests for the agent-inferred ("LLM inferred") recipient builder."""

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_recipient._filter_to_directory_users"
    )
    async def test_comma_separated_string_sets_display_name(self, mock_filter):
        """Comma-separated emails resolve to a Workload assignment with a joined display_name."""
        mock_filter.return_value = ["a@x.com", "b@x.com"]

        result = await _build_llm_recipient("a@x.com, b@x.com")

        assert result is not None
        assert result.value == "a@x.com"
        assert result.values == ["a@x.com", "b@x.com"]
        assert result.type == TaskRecipientType.WORKLOAD
        # Regression guard: display_name drives the trace `assignedTo` attribute.
        assert result.display_name == "a@x.com, b@x.com"

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_recipient._filter_to_directory_users"
    )
    async def test_list_input_resolves(self, mock_filter):
        """A list of emails is accepted and resolved."""
        mock_filter.return_value = ["a@x.com"]

        result = await _build_llm_recipient(["a@x.com"])

        assert result is not None
        assert result.value == "a@x.com"
        assert result.values == ["a@x.com"]
        assert result.type == TaskRecipientType.WORKLOAD
        assert result.display_name == "a@x.com"

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_recipient._filter_to_directory_users"
    )
    async def test_invalid_email_format_returns_none(self, mock_filter):
        """A value with no valid email format fails closed before any directory lookup."""
        result = await _build_llm_recipient("not-an-email")

        assert result is None
        mock_filter.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_string_non_list_returns_none(self):
        """Unsupported raw types (e.g. None, int) fail closed."""
        assert await _build_llm_recipient(None) is None
        assert await _build_llm_recipient(123) is None

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_recipient._filter_to_directory_users"
    )
    async def test_unresolved_directory_users_fails_closed(self, mock_filter):
        """A well-formed email that resolves to no tenant user leaves the task unassigned."""
        mock_filter.return_value = []

        result = await _build_llm_recipient("ghost@x.com")

        assert result is None


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
