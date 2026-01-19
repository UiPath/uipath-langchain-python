"""Tests for AgentsRuntimeFactory conversational agent support."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from uipath_agents._cli.runtime.factory import (
    AgentsRuntimeFactory,
)


class TestConversationalAgentInputSchema:
    """Test cases for the _get_conversational_agent_input_schema method."""

    def _resolve_ref(
        self, schema: dict[str, Any], obj: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve a $ref to its definition, handling anyOf wrappers."""
        # Handle anyOf (e.g., from Optional/Union types)
        if "anyOf" in obj:
            for option in obj["anyOf"]:
                if "$ref" in option:
                    ref_name = option["$ref"].split("/")[-1]
                    return schema["$defs"][ref_name]
        # Handle direct $ref
        if "$ref" in obj:
            ref_name = obj["$ref"].split("/")[-1]
            return schema["$defs"][ref_name]
        return obj

    def test_schema_is_object_type(self):
        """Schema should be an object type."""
        factory = AgentsRuntimeFactory(MagicMock())
        assert factory._get_conversational_agent_input_schema()["type"] == "object"

    def test_schema_allows_additional_properties(self):
        """Schema should allow additional properties for flexibility."""
        factory = AgentsRuntimeFactory(MagicMock())
        assert (
            factory._get_conversational_agent_input_schema()["additionalProperties"]
            is True
        )

    def test_schema_has_messages_property(self):
        """Schema should have a messages array property."""
        factory = AgentsRuntimeFactory(MagicMock())
        properties = factory._get_conversational_agent_input_schema()["properties"]
        assert "messages" in properties
        assert properties["messages"]["type"] == "array"

    def test_schema_has_user_settings_property(self):
        """Schema should have a userSettings object property."""
        factory = AgentsRuntimeFactory(MagicMock())
        schema = factory._get_conversational_agent_input_schema()
        properties = schema["properties"]

        assert "userSettings" in properties

        user_settings = self._resolve_ref(schema, properties["userSettings"])
        assert user_settings["type"] == "object"

    def test_user_settings_has_expected_fields(self):
        """UserSettings should have name, email, role, department, company, country, timezone."""
        factory = AgentsRuntimeFactory(MagicMock())
        schema = factory._get_conversational_agent_input_schema()

        user_settings = self._resolve_ref(schema, schema["properties"]["userSettings"])
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


class TestLoadAgentDefinition:
    """Test cases for _load_agent_definition method."""

    @pytest.fixture
    def factory(self):
        """Fixture for AgentsRuntimeFactory with mocked context."""
        mock_context = MagicMock()
        mock_context.resume = False
        mock_context.trace_manager = None
        return AgentsRuntimeFactory(mock_context)

    def test_sets_input_schema_for_conversational_agent(self, factory, tmp_path):
        """Should set the conversational input schema on the agent definition."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.input_schema = {"original": "schema"}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch("uipath_agents._cli.runtime.factory._prepare_agent_run_files"),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            assert (
                result.input_schema == factory._get_conversational_agent_input_schema()
            )

    def test_does_not_set_schema_for_regular_agent(self, factory, tmp_path):
        """Should NOT set conversational schema for regular agents."""
        original_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
        }
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = False
        mock_agent_def.input_schema = original_schema

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch("uipath_agents._cli.runtime.factory._prepare_agent_run_files"),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            # Input schema should remain unchanged for regular agents
            assert result.input_schema == original_schema

    def test_overwrites_existing_input_schema(self, factory, tmp_path):
        """Should overwrite any existing input schema for conversational agents."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.input_schema = {
            "type": "object",
            "properties": {"custom_field": {"type": "string"}},
        }

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch("uipath_agents._cli.runtime.factory._prepare_agent_run_files"),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            # Should be replaced with the conversational schema
            assert (
                result.input_schema == factory._get_conversational_agent_input_schema()
            )
            assert "custom_field" not in result.input_schema.get("properties", {})
