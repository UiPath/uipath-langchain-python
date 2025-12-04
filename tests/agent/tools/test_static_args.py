"""Tests for static_args.py module."""

from unittest.mock import MagicMock

import pytest
from uipath.agent.models.agent import (
    AgentIntegrationToolParameter,
    AgentIntegrationToolProperties,
    AgentIntegrationToolResourceConfig,
    BaseAgentResourceConfig,
)
from uipath.platform.connections import Connection

from uipath_langchain.agent.tools.static_args import (
    apply_static_args,
    resolve_integration_static_args,
    resolve_static_args,
)


class TestResolveStaticArgs:
    """Test cases for resolve_static_args function."""

    @pytest.fixture
    def connection(self):
        """Common connection object used by tests."""
        return Connection(
            id="test-connection-id", name="Test Connection", element_instance_id=12345
        )

    @pytest.fixture
    def integration_properties_factory(self, connection):
        """Factory for creating integration tool properties."""

        def _create_properties(parameters=None):
            return AgentIntegrationToolProperties(
                method="POST",
                tool_path="/api/test",
                object_name="test_object",
                tool_display_name="Test Tool",
                tool_description="Test tool description",
                connection=connection,
                parameters=parameters or [],
            )

        return _create_properties

    @pytest.fixture
    def integration_resource_factory(self, integration_properties_factory):
        """Factory for creating integration resource config."""

        def _create_resource(parameters=None):
            properties = integration_properties_factory(parameters)
            return AgentIntegrationToolResourceConfig(
                name="test_tool",
                description="Test tool",
                properties=properties,
                input_schema={},
            )

        return _create_resource

    def test_resolve_static_args_with_integration_resource(
        self, integration_resource_factory
    ):
        """Test resolve_static_args with AgentIntegrationToolResourceConfig."""
        parameters = [
            AgentIntegrationToolParameter(
                name="static_param",
                type="string",
                field_variant="static",
                field_location="body",
                value="static_value",
            )
        ]
        resource = integration_resource_factory(parameters)
        agent_input = {"input_arg": "input_value"}

        result = resolve_static_args(resource, agent_input)

        assert result == {"static_param": "static_value"}

    def test_resolve_static_args_with_unknown_resource_type(self):
        """Test resolve_static_args with unknown resource type returns empty dict."""
        mock_resource = MagicMock(spec=BaseAgentResourceConfig)
        agent_input = {"input_arg": "input_value"}

        result = resolve_static_args(mock_resource, agent_input)

        assert result == {}


class TestResolveIntegrationStaticArgs:
    """Test cases for resolve_integration_static_args function."""

    def test_resolve_with_static_values(self):
        """Test resolving parameters with static values."""
        parameters = [
            AgentIntegrationToolParameter(
                name="connection_id",
                type="string",
                field_variant="static",
                field_location="body",
                value="12345",
            ),
            AgentIntegrationToolParameter(
                name="timeout",
                type="integer",
                field_variant="static",
                field_location="body",
                value=30,
            ),
            AgentIntegrationToolParameter(
                name="config",
                type="object",
                field_variant="static",
                field_location="body",
                value={"enabled": True, "retries": 3},
            ),
        ]
        agent_input = {"user_input": "test"}

        result = resolve_integration_static_args(parameters, agent_input)

        expected = {
            "connection_id": "12345",
            "timeout": 30,
            "config": {"enabled": True, "retries": 3},
        }
        assert result == expected

    def test_resolve_with_input_arg_values(self):
        """Test resolving parameters with input argument values."""
        parameters = [
            AgentIntegrationToolParameter(
                name="user_id",
                type="string",
                field_variant="argument",
                field_location="body",
                value="{{userId}}",
            ),
            AgentIntegrationToolParameter(
                name="query",
                type="string",
                field_variant="argument",
                field_location="body",
                value="{{searchQuery}}",
            ),
        ]
        agent_input = {
            "userId": "user123",
            "searchQuery": "test search",
            "unused_arg": "not_used",
        }

        result = resolve_integration_static_args(parameters, agent_input)

        expected = {"user_id": "user123", "query": "test search"}
        assert result == expected

    def test_resolve_with_mixed_static_and_argument_values(self):
        """Test resolving parameters with both static and argument values."""
        parameters = [
            AgentIntegrationToolParameter(
                name="api_key",
                type="string",
                field_variant="static",
                field_location="body",
                value="secret_key",
            ),
            AgentIntegrationToolParameter(
                name="user_id",
                type="string",
                field_variant="argument",
                field_location="body",
                value="{{userId}}",
            ),
            AgentIntegrationToolParameter(
                name="version",
                type="string",
                field_variant="static",
                field_location="body",
                value="v1",
            ),
        ]
        agent_input = {"userId": "user456"}

        result = resolve_integration_static_args(parameters, agent_input)

        expected = {"api_key": "secret_key", "user_id": "user456", "version": "v1"}
        assert result == expected

    def test_resolve_skips_none_values(self):
        """Test that None values are skipped in the result."""
        parameters = [
            AgentIntegrationToolParameter(
                name="existing_param",
                type="string",
                field_variant="argument",
                field_location="body",
                value="{{existingArg}}",
            ),
            AgentIntegrationToolParameter(
                name="missing_param",
                type="string",
                field_variant="argument",
                field_location="body",
                value="{{missingArg}}",
            ),
        ]
        agent_input = {"existingArg": "exists"}

        result = resolve_integration_static_args(parameters, agent_input)

        assert result == {"existing_param": "exists"}
        assert "missing_param" not in result

    def test_resolve_with_invalid_argument_format_raises_error(self):
        """Test that invalid argument format raises ValueError."""
        parameters = [
            AgentIntegrationToolParameter(
                name="invalid_param",
                type="string",
                field_variant="argument",
                field_location="body",
                value="invalid_format",
            )
        ]

        with pytest.raises(ValueError, match="Parameter value must be in the format"):
            resolve_integration_static_args(parameters, {})

    def test_resolve_with_malformed_argument_braces(self):
        """Test various malformed argument brace patterns."""
        test_cases = [
            "{missing_closing",
            "missing_opening}",
            "{{missing_closing}",
            "{missing_opening}}",
            "no_braces_at_all",
        ]

        for invalid_value in test_cases:
            parameters = [
                AgentIntegrationToolParameter(
                    name="test_param",
                    type="string",
                    field_variant="argument",
                    field_location="body",
                    value=invalid_value,
                )
            ]

            with pytest.raises(ValueError):
                resolve_integration_static_args(parameters, {})


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
