"""Tests for escalation_tool.py metadata."""

import pytest
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationChannelProperties,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    StandardRecipient,
)

from uipath_langchain.agent.tools.escalation_tool import create_escalation_tool


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

    def test_escalation_tool_has_metadata(self, escalation_resource):
        """Test that escalation tool has metadata dict."""
        tool = create_escalation_tool(escalation_resource)

        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    def test_escalation_tool_metadata_has_tool_type(self, escalation_resource):
        """Test that metadata contains tool_type for span detection."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "escalation"

    def test_escalation_tool_metadata_has_display_name(self, escalation_resource):
        """Test that metadata contains display_name from app_name."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "ApprovalApp"

    def test_escalation_tool_metadata_has_channel_type(self, escalation_resource):
        """Test that metadata contains channel_type for span attributes."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["channel_type"] == "actionCenter"

    def test_escalation_tool_metadata_has_assignee(self, escalation_resource):
        """Test that metadata contains assignee when recipient is USER_EMAIL."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["assignee"] == "user@example.com"

    def test_escalation_tool_metadata_assignee_none_when_no_recipients(
        self, escalation_resource_no_recipient
    ):
        """Test that assignee is None when no recipients configured."""
        tool = create_escalation_tool(escalation_resource_no_recipient)
        assert tool.metadata is not None
        assert tool.metadata["assignee"] is None
