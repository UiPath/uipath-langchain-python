"""Tests for agent configuration loading."""

import json

import pytest
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

from uipath_agents._cli.utils import load_agent_configuration


@pytest.fixture
def valid_agent_config():
    """Fixture providing a valid agent configuration dictionary."""
    return {
        "id": "test-agent",
        "name": "Test Agent",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "{{input}}"},
        ],
        "settings": {
            "engine": "azure_openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "input_schema": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
        },
        "output_schema": {"type": "object"},
        "resources": [],
    }


@pytest.fixture
def temp_agent_file(tmp_path, valid_agent_config):
    """Fixture creating a temporary agent.json file."""
    agent_file = tmp_path / "agent.json"
    agent_file.write_text(json.dumps(valid_agent_config))
    return agent_file


class TestLoadAgentConfiguration:
    """Tests for load_agent_configuration function."""

    def test_loads_valid_configuration(self, temp_agent_file):
        """Test successful loading of valid agent configuration."""
        result = load_agent_configuration(temp_agent_file)

        assert result.id == "test-agent"
        assert result.name == "Test Agent"
        assert len(result.messages) == 2
        assert result.settings.model == "gpt-4"
        assert result.settings.temperature == 0.7

    def test_raises_when_file_not_found(self, tmp_path):
        """Test that missing file raises AgentStartupError with FILE_NOT_FOUND."""
        non_existent_file = tmp_path / "missing_agent.json"

        with pytest.raises(AgentStartupError) as exc_info:
            load_agent_configuration(non_existent_file)

        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.FILE_NOT_FOUND
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.USER

    def test_raises_on_invalid_json(self, tmp_path):
        """Test that invalid JSON raises AgentStartupError with INVALID_AGENT_CONFIG."""
        invalid_file = tmp_path / "agent.json"
        invalid_file.write_text("{invalid json content")

        with pytest.raises(AgentStartupError) as exc_info:
            load_agent_configuration(invalid_file)

        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.INVALID_AGENT_CONFIG
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM

    def test_raises_on_schema_validation_failure(self, tmp_path):
        """Test that schema validation failure raises AgentStartupError with INVALID_AGENT_CONFIG."""
        invalid_config = {
            "id": "test",
            "name": "Test",
            "messages": [],
            "settings": {
                "engine": "invalid_engine",
                "model": "gpt-4",
            },
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(invalid_config))

        with pytest.raises(AgentStartupError) as exc_info:
            load_agent_configuration(agent_file)

        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.INVALID_AGENT_CONFIG
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM
        assert "agent.json failed schema validation" in exc_info.value.error_info.detail

    def test_raises_on_missing_required_fields(self, tmp_path):
        """Test that missing required fields raises AgentStartupError with INVALID_AGENT_CONFIG."""
        incomplete_config = {"id": "test"}
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(incomplete_config))

        with pytest.raises(AgentStartupError) as exc_info:
            load_agent_configuration(agent_file)

        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.INVALID_AGENT_CONFIG
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM

    def test_validation_error_includes_details(self, tmp_path):
        """Test that validation errors include detailed error information in the detail field."""
        invalid_config = {
            "id": "test",
            "name": "Test",
            "messages": [],
            "settings": {"engine": "azure_openai", "model": 123},
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(invalid_config))

        with pytest.raises(AgentStartupError) as exc_info:
            load_agent_configuration(agent_file)

        assert "agent.json failed schema validation" in exc_info.value.error_info.detail

    def test_loads_configuration_with_resources(self, tmp_path, valid_agent_config):
        """Test loading configuration with resources defined."""
        valid_agent_config["resources"] = [
            {
                "type": "tool",
                "name": "calculator",
                "description": "A calculator tool",
                "$resourceType": "custom",
            }
        ]
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(valid_agent_config))

        result = load_agent_configuration(agent_file)

        assert len(result.resources) == 1
        # Resources are parsed as Pydantic models, access via attributes
        resource = result.resources[0]
        assert hasattr(resource, "name") or hasattr(resource, "type")

    def test_loads_configuration_with_complex_messages(
        self, tmp_path, valid_agent_config
    ):
        """Test loading configuration with complex message templates."""
        valid_agent_config["messages"] = [
            {
                "role": "system",
                "content": "Process {{data.field1}} and {{data.field2}}",
            },
            {"role": "user", "content": "Task: {{task}}"},
            {"role": "system", "content": "I will help with {{task}}"},
        ]
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(valid_agent_config))

        result = load_agent_configuration(agent_file)

        assert len(result.messages) == 3
        assert "{{data.field1}}" in result.messages[0].content
        assert result.messages[2].role == "system"

    def test_loads_configuration_with_custom_settings(
        self, tmp_path, valid_agent_config
    ):
        """Test loading configuration with various LLM settings."""
        valid_agent_config["settings"] = {
            "engine": "azure_openai",
            "model": "gpt-4-turbo",
            "temperature": 0.3,
            "max_tokens": 4096,
            "top_p": 0.9,
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(valid_agent_config))

        result = load_agent_configuration(agent_file)

        assert result.settings.model == "gpt-4-turbo"
        assert result.settings.temperature == 0.3
        assert result.settings.max_tokens == 4096

    def test_handles_unicode_content(self, tmp_path, valid_agent_config):
        """Test loading configuration with unicode characters."""
        valid_agent_config["messages"][0]["content"] = "Process data with émojis 🚀"
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(
            json.dumps(valid_agent_config, ensure_ascii=False), encoding="utf-8"
        )

        result = load_agent_configuration(agent_file)

        assert "🚀" in result.messages[0].content

    def test_preserves_input_output_schemas(self, tmp_path, valid_agent_config):
        """Test that input and output schemas are preserved correctly."""
        valid_agent_config["input_schema"] = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        }
        valid_agent_config["output_schema"] = {
            "type": "object",
            "properties": {"results": {"type": "array"}},
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(valid_agent_config))

        result = load_agent_configuration(agent_file)

        assert "query" in result.input_schema["properties"]
        assert "results" in result.output_schema["properties"]

    def test_normalizes_content_token_types_case_insensitive(
        self, tmp_path, valid_agent_config
    ):
        """Test that content token types are normalized by decapitalizing first letter."""
        valid_agent_config["messages"] = [
            {
                "role": "system",
                "content": "Hello World",
                "contentTokens": [
                    {"type": "SimpleText", "rawString": "Hello "},
                    {"type": "Variable", "rawString": "{{name}}"},
                    {"type": "simpleText", "rawString": " World"},
                ],
            },
            {
                "role": "user",
                "content": "Input",
                "contentTokens": [
                    {"type": "Expression", "rawString": "{{input}}"},
                ],
            },
        ]
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(valid_agent_config))

        result = load_agent_configuration(agent_file)

        assert len(result.messages) == 2
        assert result.messages[0].content_tokens is not None
        assert len(result.messages[0].content_tokens) == 3
        assert result.messages[0].content_tokens[0].type.value == "simpleText"
        assert result.messages[0].content_tokens[1].type.value == "variable"
        assert result.messages[0].content_tokens[2].type.value == "simpleText"
        assert result.messages[1].content_tokens is not None
        assert result.messages[1].content_tokens[0].type.value == "expression"
