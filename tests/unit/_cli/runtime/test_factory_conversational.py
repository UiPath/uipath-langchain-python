"""Tests for AgentsRuntimeFactory conversational agent support."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from uipath_agents._cli.runtime.factory import (
    AgentsRuntimeFactory,
    conversational_agent_input_schema,
)


class TestConversationalAgentInputSchema:
    """Test cases for the conversational_agent_input_schema constant."""

    def test_schema_is_object_type(self):
        """Schema should be an object type."""
        assert conversational_agent_input_schema["type"] == "object"

    def test_schema_allows_additional_properties(self):
        """Schema should allow additional properties for flexibility."""
        assert conversational_agent_input_schema["additionalProperties"] is True

    def test_schema_has_messages_property(self):
        """Schema should have a messages array property."""
        properties = conversational_agent_input_schema["properties"]
        assert "messages" in properties
        assert properties["messages"]["type"] == "array"

    def test_messages_items_have_expected_structure(self):
        """Messages items should have messageId, role, and contentParts."""
        message_item = conversational_agent_input_schema["properties"]["messages"][
            "items"
        ]
        print(f">>>> message_item: {message_item}")
        props = message_item["properties"]

        assert "messageId" in props
        assert "role" in props
        assert "contentParts" in props

    def test_schema_has_user_settings_property(self):
        """Schema should have a userSettings object property."""
        properties = conversational_agent_input_schema["properties"]
        assert "userSettings" in properties
        assert properties["userSettings"]["type"] == "object"

    def test_user_settings_has_expected_fields(self):
        """UserSettings should have name, email, role, department, company, country, timezone."""
        user_settings = conversational_agent_input_schema["properties"]["userSettings"]
        props = user_settings["properties"]

        expected_fields = [
            "name",
            "email",
            "role",
            "department",
            "company",
            "country",
            "timezone",
        ]
        for field in expected_fields:
            assert field in props, f"Missing field: {field}"


class TestFixupConversationalAgentDefinition:
    """Test cases for _fixup_conversational_agent_definition method."""

    @pytest.fixture
    def factory(self):
        """Fixture for AgentsRuntimeFactory with mocked context."""
        mock_context = MagicMock()
        mock_context.resume = False
        mock_context.trace_manager = None
        return AgentsRuntimeFactory(mock_context)

    def test_sets_input_schema_for_conversational_agent(self, factory):
        """Should set the conversational input schema on the agent definition."""
        mock_agent_def = MagicMock()
        mock_agent_def.input_schema = {"original": "schema"}

        factory._fixup_conversational_agent_definition(mock_agent_def)

        assert mock_agent_def.input_schema == conversational_agent_input_schema

    def test_overwrites_existing_input_schema(self, factory):
        """Should overwrite any existing input schema."""
        mock_agent_def = MagicMock()
        mock_agent_def.input_schema = {
            "type": "object",
            "properties": {"custom_field": {"type": "string"}},
        }

        factory._fixup_conversational_agent_definition(mock_agent_def)

        # Should be replaced with the conversational schema
        assert mock_agent_def.input_schema == conversational_agent_input_schema
        assert "custom_field" not in mock_agent_def.input_schema.get("properties", {})


class TestAgentsRuntimeFactoryLoadGraph:
    """Test cases for _load_graph with conversational agents."""

    @pytest.fixture
    def factory(self):
        """Fixture for AgentsRuntimeFactory with mocked context."""
        mock_context = MagicMock()
        mock_context.resume = False
        mock_context.trace_manager = None
        mock_context.command = "run"
        mock_context.get_input.return_value = {
            "messages": [],
            "userSettings": {"name": "Test"},
        }
        return AgentsRuntimeFactory(mock_context)

    @pytest.mark.asyncio
    async def test_calls_fixup_for_conversational_agent(self, factory, tmp_path):
        """Should call _fixup_conversational_agent_definition for conversational agents."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.name = "Test Agent"
        mock_agent_def.settings = MagicMock()
        mock_agent_def.settings.model = "gpt-4"
        mock_agent_def.settings.max_tokens = 1000
        mock_agent_def.settings.temperature = 0.7
        mock_agent_def.settings.engine = "azure_openai"
        mock_agent_def.settings.max_iterations = 10
        mock_agent_def.input_schema = {"type": "object"}
        mock_agent_def.output_schema = {"type": "object"}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch("uipath_agents._cli.runtime.factory._prepare_agent_run_files"),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
            patch(
                "uipath_agents._cli.runtime.factory.validate_json_against_json_schema"
            ),
            patch("uipath_agents._cli.runtime.factory.build_agent_graph") as mock_build,
        ):
            mock_build.return_value = MagicMock()

            await factory._load_graph("agent.json")

            # Verify input schema was set to conversational schema
            assert mock_agent_def.input_schema == conversational_agent_input_schema

    @pytest.mark.asyncio
    async def test_does_not_call_fixup_for_regular_agent(self, factory, tmp_path):
        """Should NOT call _fixup_conversational_agent_definition for regular agents."""
        original_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
        }
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = False
        mock_agent_def.name = "Regular Agent"
        mock_agent_def.settings = MagicMock()
        mock_agent_def.settings.model = "gpt-4"
        mock_agent_def.settings.max_tokens = 1000
        mock_agent_def.settings.temperature = 0.7
        mock_agent_def.settings.engine = "azure_openai"
        mock_agent_def.settings.max_iterations = 10
        mock_agent_def.input_schema = original_schema
        mock_agent_def.output_schema = {"type": "object"}

        factory.context.get_input.return_value = {"task": "test"}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch("uipath_agents._cli.runtime.factory._prepare_agent_run_files"),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
            patch(
                "uipath_agents._cli.runtime.factory.validate_json_against_json_schema"
            ),
            patch("uipath_agents._cli.runtime.factory.build_agent_graph") as mock_build,
        ):
            mock_build.return_value = MagicMock()

            await factory._load_graph("agent.json")

            # Input schema should remain unchanged for regular agents
            assert mock_agent_def.input_schema == original_schema

    @pytest.mark.asyncio
    async def test_stores_is_conversational_in_agent_info(self, factory, tmp_path):
        """Should store is_conversational flag in agent_info for telemetry."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.name = "Conversational Agent"
        mock_agent_def.settings = MagicMock()
        mock_agent_def.settings.model = "gpt-4"
        mock_agent_def.settings.max_tokens = 1000
        mock_agent_def.settings.temperature = 0.7
        mock_agent_def.settings.engine = "azure_openai"
        mock_agent_def.settings.max_iterations = 10
        mock_agent_def.input_schema = {"type": "object"}
        mock_agent_def.output_schema = {"type": "object"}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch("uipath_agents._cli.runtime.factory._prepare_agent_run_files"),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
            patch(
                "uipath_agents._cli.runtime.factory.validate_json_against_json_schema"
            ),
            patch("uipath_agents._cli.runtime.factory.build_agent_graph") as mock_build,
        ):
            mock_build.return_value = MagicMock()

            await factory._load_graph("agent.json")

            # Verify agent_info was set with is_conversational
            assert factory._agent_info is not None
            assert factory._agent_info.is_conversational is True

    @pytest.mark.asyncio
    async def test_validates_input_against_conversational_schema(
        self, factory, tmp_path
    ):
        """Should validate input against the conversational schema after fixup."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.name = "Test Agent"
        mock_agent_def.settings = MagicMock()
        mock_agent_def.settings.model = "gpt-4"
        mock_agent_def.settings.max_tokens = None
        mock_agent_def.settings.temperature = None
        mock_agent_def.settings.engine = None
        mock_agent_def.settings.max_iterations = None
        mock_agent_def.input_schema = {"original": "schema"}
        mock_agent_def.output_schema = {}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch("uipath_agents._cli.runtime.factory._prepare_agent_run_files"),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
            patch(
                "uipath_agents._cli.runtime.factory.validate_json_against_json_schema"
            ) as mock_validate,
            patch("uipath_agents._cli.runtime.factory.build_agent_graph") as mock_build,
        ):
            mock_build.return_value = MagicMock()

            await factory._load_graph("agent.json")

            # Validate should be called with the conversational schema (after fixup)
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args[0]
            # First arg is the schema
            assert call_args[0] == conversational_agent_input_schema
