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
        assert factory._get_conversational_agent_input_schema(None)["type"] == "object"

    def test_schema_has_messages_property(self):
        """Schema should have a messages array property."""
        factory = AgentsRuntimeFactory(MagicMock())
        properties = factory._get_conversational_agent_input_schema(None)["properties"]
        assert "messages" in properties
        assert properties["messages"]["type"] == "array"

    def test_schema_has_uipath__user_settings_property(self):
        """Schema should have a uipath__user_settings object property."""
        factory = AgentsRuntimeFactory(MagicMock())
        schema = factory._get_conversational_agent_input_schema(None)
        properties = schema["properties"]

        assert "uipath__user_settings" in properties

        user_settings = self._resolve_ref(schema, properties["uipath__user_settings"])
        assert user_settings["type"] == "object"

    def test_uipath__user_settings_has_expected_fields(self):
        """uipath__user_settings should have name, email, role, department, company, country, timezone."""
        factory = AgentsRuntimeFactory(MagicMock())
        schema = factory._get_conversational_agent_input_schema(None)

        user_settings = self._resolve_ref(
            schema, schema["properties"]["uipath__user_settings"]
        )
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

    def test_none_existing_schema_returns_default_schema(self):
        """When existing_output_schema is None, should return default schema."""
        factory = AgentsRuntimeFactory(MagicMock())
        result = factory._get_conversational_agent_input_schema(None)

        assert "properties" in result
        assert len(result["properties"]) == 2
        assert "messages" in result["properties"]
        assert "uipath__user_settings" in result["properties"]

    def test_empty_dict_existing_schema_returns_default_schema(self):
        """When existing_output_schema is empty dict, should return default schema."""
        factory = AgentsRuntimeFactory(MagicMock())
        result = factory._get_conversational_agent_input_schema({})

        assert "properties" in result
        assert len(result["properties"]) == 2
        assert "messages" in result["properties"]
        assert "uipath__user_settings" in result["properties"]

    def test_merges_existing_schema_properties(self):
        """Should merge existing schema properties with system fields."""
        factory = AgentsRuntimeFactory(MagicMock())

        existing_schema = {
            "type": "object",
            "title": "Custom Agent Input",
            "description": "User-defined schema",
            "properties": {
                "custom_field": {"type": "string", "description": "A custom field"},
                "another_field": {"type": "number"},
            },
            "required": ["custom_field"],
            "additionalProperties": False,
        }

        result = factory._get_conversational_agent_input_schema(existing_schema)

        # Default low-code conversational agent system fields should be present
        assert "messages" in result["properties"]
        assert "uipath__user_settings" in result["properties"]

        # Existing schema properties should be preserved
        assert "custom_field" in result["properties"]
        assert "another_field" in result["properties"]
        assert result["properties"]["custom_field"]["type"] == "string"

        # Metadata should be preserved
        assert result["title"] == "Custom Agent Input"
        assert result["description"] == "User-defined schema"
        assert result["additionalProperties"] is False

        # Existing schema required fields should be preserved
        assert "custom_field" in result["required"]
        assert "another_field" not in result["required"]
        assert "messages" not in result["required"]
        assert "uipath__user_settings" not in result["required"]


class TestConversationalAgentOutputSchema:
    """Test cases for the _get_conversational_agent_output_schema method."""

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
        assert factory._get_conversational_agent_output_schema(None)["type"] == "object"

    def test_schema_has_uipath__agent_response_messages_property(self):
        """Schema should have a uipath__agent_response_messages array property."""
        factory = AgentsRuntimeFactory(MagicMock())
        result = factory._get_conversational_agent_output_schema(None)
        properties = result["properties"]

        assert "uipath__agent_response_messages" in properties
        assert properties["uipath__agent_response_messages"]["type"] == "array"

    def test_uipath__agent_response_messages_has_default_value(self):
        """uipath__agent_response_messages should have a default value of empty list."""
        factory = AgentsRuntimeFactory(MagicMock())
        result = factory._get_conversational_agent_output_schema(None)
        properties = result["properties"]

        assert "default" in properties["uipath__agent_response_messages"]
        assert properties["uipath__agent_response_messages"]["default"] == []

    def test_none_existing_schema_returns_default_schema(self):
        """When existing_output_schema is None, should return default schema."""
        factory = AgentsRuntimeFactory(MagicMock())
        result = factory._get_conversational_agent_output_schema(None)

        assert "properties" in result
        assert len(result["properties"]) == 1
        assert "uipath__agent_response_messages" in result["properties"]

    def test_empty_dict_existing_schema_returns_default_schema(self):
        """When existing_output_schema is empty dict, should return default schema."""
        factory = AgentsRuntimeFactory(MagicMock())
        result = factory._get_conversational_agent_output_schema({})

        assert "properties" in result
        assert len(result["properties"]) == 1
        assert "uipath__agent_response_messages" in result["properties"]

    def test_merges_existing_schema_properties(self):
        """Should merge existing schema properties with system fields."""
        factory = AgentsRuntimeFactory(MagicMock())

        existing_schema = {
            "type": "object",
            "title": "Custom Agent Output",
            "description": "User-defined output schema",
            "properties": {
                "output_string": {
                    "type": "string",
                    "description": "A custom output field",
                },
                "output_int": {"type": "integer"},
            },
            "required": ["output_string"],
            "additionalProperties": False,
        }

        result = factory._get_conversational_agent_output_schema(existing_schema)

        # System field should be present
        assert "uipath__agent_response_messages" in result["properties"]

        # Existing schema properties should be preserved
        assert len(result["properties"]) == 3
        assert "output_string" in result["properties"]
        assert "output_int" in result["properties"]
        assert result["properties"]["output_string"]["type"] == "string"
        assert result["properties"]["output_int"]["type"] == "integer"

        # Metadata should be preserved
        assert result["title"] == "Custom Agent Output"
        assert result["description"] == "User-defined output schema"
        assert result["additionalProperties"] is False


class TestLoadAgentDefinition:
    """Test cases for _load_agent_definition method."""

    @pytest.fixture
    def factory(self):
        """Fixture for AgentsRuntimeFactory with mocked context."""
        mock_context = MagicMock()
        mock_context.resume = False
        mock_context.trace_manager = None
        # Patch _prepare_agent_execution_contract since it's called in __init__
        with patch(
            "uipath_agents._cli.runtime.factory._prepare_agent_execution_contract"
        ):
            return AgentsRuntimeFactory(mock_context)

    def test_sets_input_schema_for_conversational_agent(self, factory, tmp_path):
        """Should set the conversational input schema on the agent definition."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.input_schema = {"original": "schema"}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            assert (
                result.input_schema
                == factory._get_conversational_agent_input_schema(
                    mock_agent_def.input_schema
                )
            )

    def test_does_not_set_schema_for_non_conversational_agent(self, factory, tmp_path):
        """Should NOT set conversational schema for non-conversational agents."""
        original_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
        }
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = False
        mock_agent_def.input_schema = original_schema

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            # Input schema should remain unchanged for regular agents
            assert result.input_schema == original_schema

    def test_merges_existing_input_schema(self, factory, tmp_path):
        """Should merge existing input schema with system fields for conversational agents."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        existing_schema = {
            "type": "object",
            "properties": {"custom_field": {"type": "string"}},
            "required": ["custom_field"],
        }
        mock_agent_def.input_schema = existing_schema

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            assert (
                result.input_schema
                == factory._get_conversational_agent_input_schema(existing_schema)
            )

            # Default low-code conversational agent system fields should be present
            assert "messages" in result.input_schema["properties"]
            assert "uipath__user_settings" in result.input_schema["properties"]

            # Existing schema properties should be preserved
            assert "custom_field" in result.input_schema["properties"]
            assert result.input_schema["properties"]["custom_field"]["type"] == "string"

            # Existing schema required fields should be preserved
            assert "custom_field" in result.input_schema["required"]
            assert "messages" not in result.input_schema["required"]
            assert "uipath__user_settings" not in result.input_schema["required"]

    def test_sets_output_schema_for_conversational_agent(self, factory, tmp_path):
        """Should set the conversational output schema on the agent definition."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.output_schema = {"original": "schema"}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            assert (
                result.output_schema
                == factory._get_conversational_agent_output_schema(
                    mock_agent_def.output_schema
                )
            )

    def test_does_not_set_output_schema_for_non_conversational_agent(
        self, factory, tmp_path
    ):
        """Should NOT set conversational output schema for regular agents."""
        original_output_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = False
        mock_agent_def.input_schema = {}
        mock_agent_def.output_schema = original_output_schema

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            # Output schema should remain unchanged for regular agents
            assert result.output_schema == original_output_schema
            assert (
                "uipath__agent_response_messages"
                not in result.output_schema["properties"]
            )

    def test_currently_ignores_output_schema_for_conversational_agent(
        self, factory, tmp_path
    ):
        """Should merge existing output schema with system fields for conversational agents."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.input_schema = {}
        existing_output_schema = {
            "type": "object",
            "properties": {"custom_field": {"type": "string"}},
            "required": ["custom_field"],
        }
        mock_agent_def.output_schema = existing_output_schema

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            # System field should be present
            assert (
                "uipath__agent_response_messages" in result.output_schema["properties"]
            )

            # Any other schema properties should not exist
            assert len(result.output_schema["properties"]) == 1
            assert "custom_field" not in result.output_schema["properties"]

    def test_both_input_and_output_schemas_set_for_conversational_agent(
        self, factory, tmp_path
    ):
        """Both input and output schemas should be set for conversational agents."""
        mock_agent_def = MagicMock()
        mock_agent_def.is_conversational = True
        mock_agent_def.input_schema = {}
        mock_agent_def.output_schema = {}

        with (
            patch("uipath_agents._cli.runtime.factory.Path.cwd", return_value=tmp_path),
            patch(
                "uipath_agents._cli.runtime.factory.load_agent_configuration",
                return_value=mock_agent_def,
            ),
        ):
            result = factory._load_agent_definition("agent.json")

            # Input schema should have conversational fields
            assert "messages" in result.input_schema["properties"]
            assert "uipath__user_settings" in result.input_schema["properties"]

            # Output schema should have conversational fields
            assert (
                "uipath__agent_response_messages" in result.output_schema["properties"]
            )
