"""Tests for flow-control tool factories in agent.react.tools.tools."""

from pydantic import BaseModel, Field
from uipath.agent.react import SET_CONVERSATIONAL_OUTPUT_TOOL

from uipath_langchain.agent.react.tools.tools import (
    create_set_conversational_output_tool,
)


class _FullOutputSchema(BaseModel):
    uipath__agent_response_messages: list = Field(default_factory=list)
    handoff_target: str = "none"
    ready_for_handoff: bool = False
    urgency: str | None = None


class TestCreateSetConversationalOutputTool:
    def test_tool_name_matches_registry(self):
        tool = create_set_conversational_output_tool(_FullOutputSchema)
        assert tool.name == SET_CONVERSATIONAL_OUTPUT_TOOL.name

    def test_tool_description_matches_registry(self):
        tool = create_set_conversational_output_tool(_FullOutputSchema)
        assert tool.description == SET_CONVERSATIONAL_OUTPUT_TOOL.description

    def test_args_schema_strips_response_messages_field(self):
        tool = create_set_conversational_output_tool(_FullOutputSchema)
        fields = tool.args_schema.model_fields
        assert "uipath__agent_response_messages" not in fields
        assert "handoff_target" in fields
        assert "ready_for_handoff" in fields
        assert "urgency" in fields

    def test_args_schema_accepts_partial_payload(self):
        """Validates the "N/A"-style placeholder workflow — the schema must
        accept partial payloads where optional fields are omitted."""
        tool = create_set_conversational_output_tool(_FullOutputSchema)
        validated = tool.args_schema.model_validate(
            {"handoff_target": "billing", "ready_for_handoff": True}
        )
        assert validated.handoff_target == "billing"
        assert validated.ready_for_handoff is True
        assert validated.urgency is None
