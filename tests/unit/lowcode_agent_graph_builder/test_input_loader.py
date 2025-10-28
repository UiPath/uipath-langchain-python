"""Tests for input loading and validation."""

import json
from pathlib import Path
from typing import Any

import pytest

from uipath_lowcode.lowcode_agent_graph_builder.exceptions import (
    ConfigurationError,
    InputValidationError,
)
from uipath_lowcode.lowcode_agent_graph_builder.input_loader import (
    load_agent_configuration,
    load_input_arguments,
)


def create_valid_agent_config(**overrides):
    """Create a valid minimal agent configuration for testing."""
    config = {
        "id": "test-agent",
        "name": "Test Agent",
        "messages": [{"role": "system", "content": "test"}],
        "settings": {
            "engine": "azure_openai",
            "model": "gpt-4",
            "maxTokens": 4096,
            "temperature": 0.7,
        },
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object"},
        "resources": [],
    }
    config.update(overrides)
    return config


class TestLoadAgentConfiguration:
    """Test loading agent configuration from file."""

    def test_missing_file(self, tmp_path: Path):
        """Test error when agent.json doesn't exist."""
        non_existent = tmp_path / "agent.json"
        with pytest.raises(ConfigurationError, match="not found"):
            load_agent_configuration(non_existent)

    def test_invalid_json(self, tmp_path: Path):
        """Test error when file contains invalid JSON."""
        config_file = tmp_path / "agent.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises((ConfigurationError, InputValidationError)):
            load_agent_configuration(config_file)

    def test_valid_minimal_config(self, tmp_path: Path):
        """Test loading valid minimal configuration."""
        config_file = tmp_path / "agent.json"
        config = create_valid_agent_config()
        config_file.write_text(json.dumps(config))

        result = load_agent_configuration(config_file)
        assert result.settings.model == "gpt-4"
        assert result.settings.temperature == 0.7


class TestLoadInputArguments:
    """Test loading and validating input arguments."""

    def test_none_input(self):
        """Test None input returns empty dict."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        result = load_input_arguments(schema, None)
        assert result == {}

    def test_empty_string_input(self):
        """Test empty string input returns empty dict."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        result = load_input_arguments(schema, "")
        assert result == {}

    def test_dict_input_valid(self):
        """Test valid dict input."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = load_input_arguments(schema, {"name": "Alice"})
        assert result["name"] == "Alice"

    def test_json_string_input_valid(self):
        """Test valid JSON string input."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        result = load_input_arguments(schema, '{"count": 42}')
        assert result["count"] == 42

    def test_validation_error_missing_required(self):
        """Test validation error for missing required fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        with pytest.raises(InputValidationError, match="schema validation"):
            load_input_arguments(schema, {})

    def test_validation_error_wrong_type(self):
        """Test validation error for wrong type."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        with pytest.raises(InputValidationError):
            load_input_arguments(schema, {"count": "not a number"})

    def test_invalid_json_string(self):
        """Test error for invalid JSON string."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        with pytest.raises((ConfigurationError, InputValidationError)):
            load_input_arguments(schema, "{ invalid json }")

    def test_nested_object_validation(self):
        """Test validation of nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "required": ["user"],
        }
        result = load_input_arguments(schema, {"user": {"name": "Bob"}})
        assert result["user"]["name"] == "Bob"
