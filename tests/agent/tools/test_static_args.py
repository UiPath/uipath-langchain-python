"""Tests for static_args.py module."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolStaticArgumentProperties,
    BaseAgentResourceConfig,
)

from uipath_langchain.agent.tools.static_args import (
    apply_static_args,
    apply_static_argument_properties_to_schema,
    resolve_static_args,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)


class TestResolveStaticArgs:
    """Test cases for resolve_static_args function."""

    def test_resolve_static_args_with_argument_properties(self):
        """Test resolve_static_args with an object that has argument_properties."""

        class ResourceWithProps:
            argument_properties = {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
            }

        result = resolve_static_args(ResourceWithProps(), {"unused": "input"})

        assert result == {"$['host']": "api.example.com"}

    def test_resolve_static_args_with_static_values_of_different_types(self):
        """Test resolve_static_args resolves string, integer, and object static values."""

        class ResourceWithProps:
            argument_properties = {
                "$['connection_id']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="12345"
                ),
                "$['timeout']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value=30
                ),
                "$['config']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value={"enabled": True, "retries": 3}
                ),
            }

        result = resolve_static_args(ResourceWithProps(), {"unused": "input"})

        assert result == {
            "$['connection_id']": "12345",
            "$['timeout']": 30,
            "$['config']": {"enabled": True, "retries": 3},
        }

    def test_resolve_static_args_with_argument_properties_extracts_from_agent_input(
        self,
    ):
        """Test resolve_static_args resolves AgentToolArgumentArgumentProperties from agent_input."""

        class ResourceWithProps:
            argument_properties = {
                "$['user_id']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="userId"
                ),
                "$['query']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="searchQuery"
                ),
            }

        agent_input = {
            "userId": "user123",
            "searchQuery": "test search",
            "unused_arg": "not_used",
        }

        result = resolve_static_args(ResourceWithProps(), agent_input)

        assert result == {
            "$['user_id']": "user123",
            "$['query']": "test search",
        }

    def test_resolve_static_args_with_mixed_static_and_argument_properties(self):
        """Test resolve_static_args with both static and argument properties."""

        class ResourceWithProps:
            argument_properties = {
                "$['api_key']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="secret_key"
                ),
                "$['user_id']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="userId"
                ),
                "$['version']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="v1"
                ),
            }

        agent_input = {"userId": "user456"}

        result = resolve_static_args(ResourceWithProps(), agent_input)

        assert result == {
            "$['api_key']": "secret_key",
            "$['user_id']": "user456",
            "$['version']": "v1",
        }

    def test_resolve_static_args_skips_missing_argument_values(self):
        """Test that argument properties referencing missing agent_input keys are skipped."""

        class ResourceWithProps:
            argument_properties = {
                "$['existing_param']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="existingArg"
                ),
                "$['missing_param']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="missingArg"
                ),
            }

        agent_input = {"existingArg": "exists"}

        result = resolve_static_args(ResourceWithProps(), agent_input)

        assert result == {"$['existing_param']": "exists"}
        assert "$['missing_param']" not in result

    def test_resolve_static_args_with_unknown_resource_type(self):
        """Test resolve_static_args with unknown resource type returns empty dict."""
        mock_resource = MagicMock(spec=BaseAgentResourceConfig)
        agent_input = {"input_arg": "input_value"}

        result = resolve_static_args(mock_resource, agent_input)

        assert result == {}


class TestApplyStaticArgs:
    """Test cases for apply_static_args function."""

    def test_apply_static_args_top_level_simple_fields(self):
        """Test applying static args to top level simple fields."""
        static_args = {"field1": "value1", "field2": 42, "field3": True}
        kwargs = {"existing_field": "existing"}

        result = apply_static_args(static_args, kwargs)

        expected = {
            "existing_field": "existing",
            "field1": "value1",
            "field2": 42,
            "field3": True,
        }
        assert result == expected

    def test_apply_static_args_top_level_objects_replace_whole(self):
        """Test applying static args to top level objects - should replace whole object."""
        static_args = {"config": {"new_setting": "new_value", "enabled": True}}
        kwargs = {"config": {"old_setting": "old_value"}}

        result = apply_static_args(static_args, kwargs)

        expected = {"config": {"new_setting": "new_value", "enabled": True}}
        assert result == expected

    def test_apply_static_args_top_level_arrays_replace_entire(self):
        """Test applying static args to top level arrays - should replace entire array."""
        static_args = {"items": ["new_item1", "new_item2"]}
        kwargs = {"items": ["old_item1", "old_item2", "old_item3"]}

        result = apply_static_args(static_args, kwargs)

        expected = {"items": ["new_item1", "new_item2"]}
        assert result == expected

    def test_apply_static_args_nested_property_in_object_two_levels(self):
        """Test applying static args to nested property in object (2 levels deep) - should replace only property."""
        static_args = {"config.database.host": "new_host"}
        kwargs = {
            "config": {
                "database": {"host": "old_host", "port": 5432},
                "cache": {"enabled": True},
            }
        }

        result = apply_static_args(static_args, kwargs)

        expected = {
            "config": {
                "database": {"host": "new_host", "port": 5432},
                "cache": {"enabled": True},
            }
        }
        assert result == expected

    def test_apply_static_args_array_element_replace_every_element(self):
        """Test applying static args to array elements - should replace every element."""
        static_args = {"users[*]": {"status": "active"}}
        kwargs = {
            "users": [
                {"id": 1, "name": "John", "status": "inactive"},
                {"id": 2, "name": "Jane", "status": "pending"},
            ]
        }

        result = apply_static_args(static_args, kwargs)

        expected = {"users": [{"status": "active"}, {"status": "active"}]}
        assert result == expected

    def test_apply_static_args_to_empty_array_replaces_with_static_value(self):
        """Test applying static args to empty array - should replace with single static value."""
        static_args = {"$['files'][*]": {"id": "uuid-123"}}
        kwargs: dict[str, Any] = {
            "files": [],
        }

        result = apply_static_args(static_args, kwargs)
        assert result == {"files": [{"id": "uuid-123"}]}

    def test_apply_static_args_nested_property_in_array_element(self):
        """Test applying static args to nested property in array element - should replace property on every object."""
        static_args = {"users[*].profile.verified": True}
        kwargs = {
            "users": [
                {
                    "id": 1,
                    "profile": {"verified": False, "email": "john@example.com"},
                },
                {
                    "id": 2,
                    "profile": {"verified": False, "email": "jane@example.com"},
                },
            ]
        }

        result = apply_static_args(static_args, kwargs)

        expected = {
            "users": [
                {"id": 1, "profile": {"verified": True, "email": "john@example.com"}},
                {"id": 2, "profile": {"verified": True, "email": "jane@example.com"}},
            ]
        }
        assert result == expected

    def test_apply_static_args_with_pydantic_models(self):
        """Test applying static args with Pydantic models in arguments."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            class InnerModel(BaseModel):
                detail: str
                count: int

            name: str
            value: InnerModel

        static_args = {"model_arg.value.detail": "static_value"}

        model_instance = TestModel(
            name="test", value=TestModel.InnerModel(detail="detail", count=123)
        )
        kwargs = {"model_arg": model_instance}

        result = apply_static_args(static_args, kwargs)

        expected = {
            "model_arg": {
                "name": "test",
                "value": {"detail": "static_value", "count": 123},
            },
        }
        assert result == expected

    def test_apply_static_args_with_list_of_pydantic_models(self):
        """Test applying static args with list of Pydantic models."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        static_args = {"models[*].processed": True}

        models = [TestModel(name="test1", value=1), TestModel(name="test2", value=2)]
        kwargs = {"models": models}

        result = apply_static_args(static_args, kwargs)

        expected = {
            "models": [
                {"name": "test1", "value": 1, "processed": True},
                {"name": "test2", "value": 2, "processed": True},
            ]
        }
        assert result == expected

    def test_apply_static_args_creates_missing_nested_structure(self):
        """Test that apply_static_args creates missing nested structure."""
        static_args = {"config.new_section.setting": "value"}

        result = apply_static_args(static_args, {})

        expected = {"config": {"new_section": {"setting": "value"}}}
        assert result == expected

    def test_apply_static_args_replace_entire_array(self):
        """Test applying static args to nested array - should replace entire array."""
        static_args = {"$['config']['allowed_ips']": ["192.168.1.1", "10.0.0.1"]}
        kwargs = {
            "config": {
                "allowed_ips": ["172.16.0.1", "172.16.0.2", "172.16.0.3"],
                "timeout": 30,
            },
            "enabled": True,
        }

        result = apply_static_args(static_args, kwargs)

        expected = {
            "config": {
                "allowed_ips": ["192.168.1.1", "10.0.0.1"],
                "timeout": 30,
            },
            "enabled": True,
        }
        assert result == expected


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    host: str
    port: int = Field(default=8080)
    api_key: str


class TestApplyStaticArgumentPropertiesToSchema:
    """Test cases for apply_static_argument_properties_to_schema function."""

    def create_test_tool(
        self, argument_properties: dict[str, AgentToolArgumentProperties]
    ) -> StructuredToolWithArgumentProperties:
        """Create a test tool for testing."""

        async def tool_fn(host: str, port: int = 8080, api_key: str = "") -> str:
            return f"{host}:{port}"

        return StructuredToolWithArgumentProperties(
            name="test_tool",
            description="A test tool",
            args_schema=SimpleInput,
            coroutine=tool_fn,
            output_type=None,
            argument_properties=argument_properties,
        )

    @pytest.fixture
    def agent_input(self) -> dict[str, Any]:
        """Common agent input for tests."""
        return {"user_id": "user123", "query": "test query"}

    def test_returns_original_tool_when_no_properties(
        self, agent_input: dict[str, Any]
    ) -> None:
        """Test that the original tool is returned when argument_properties is empty."""
        tool = self.create_test_tool({})
        result = apply_static_argument_properties_to_schema(tool, agent_input)

        assert result is tool

    def test_returns_modified_tool_with_static_properties(
        self, agent_input: dict[str, Any]
    ) -> None:
        """Test that a modified tool is returned when static properties are provided."""
        tool = self.create_test_tool(
            {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False,
                    value="api.example.com",
                ),
                "$['api_key']": AgentToolStaticArgumentProperties(
                    is_sensitive=True,
                    value="secret-key-123",
                ),
            }
        )

        result = apply_static_argument_properties_to_schema(tool, agent_input)

        # Should return a different tool instance
        assert result is not tool
        assert result.name == tool.name
        assert result.description == tool.description
        assert isinstance(result.args_schema, type(BaseModel))
        schema = result.args_schema.model_json_schema()

        assert "pre-configured" in schema["properties"]["api_key"]["description"]
        assert "api_key" not in schema["required"]
        host_def = schema["$defs"]["Host"]
        assert host_def["enum"] == ["api.example.com"]

    def test_skips_invalid_argument_properties(
        self, agent_input: dict[str, Any]
    ) -> None:
        tool = self.create_test_tool(
            {
                "$['nonexistent_field']": AgentToolStaticArgumentProperties(
                    is_sensitive=False,
                    value="test",
                ),
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False,
                    value="api.example.com",
                ),
            }
        )

        result = apply_static_argument_properties_to_schema(tool, agent_input)

        assert isinstance(result.args_schema, type(BaseModel))
        schema = result.args_schema.model_json_schema()
        host_def = schema["$defs"]["Host"]
        assert host_def["enum"] == ["api.example.com"]
        assert "nonexistent_field" not in schema["properties"]
