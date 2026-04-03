"""Tests for static_args.py module."""

from typing import Any

from langchain_core.messages import ToolCall
from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolStaticArgumentProperties,
)

from uipath_langchain.agent.tools.static_args import (
    ArgumentPropertiesMixin,
    StaticArgsHandler,
    apply_static_args,
    resolve_static_args,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    host: str
    port: int = Field(default=8080)
    api_key: str


def _create_tool(
    name: str,
    argument_properties: dict[str, AgentToolArgumentProperties],
    args_schema: type[BaseModel] = SimpleInput,
) -> StructuredToolWithArgumentProperties:
    async def tool_fn(**kwargs: Any) -> str:
        return "ok"

    return StructuredToolWithArgumentProperties(
        name=name,
        description="A test tool",
        args_schema=args_schema,
        coroutine=tool_fn,
        output_type=None,
        argument_properties=argument_properties,
    )


class EmptyInput(BaseModel):
    """Empty input schema for tests that don't need agent input."""

    pass


def _make_tool_call(name: str, args: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(name=name, args=args or {}, id="1", type="tool_call")


class TestStaticArgsHandler:
    """Test cases for StaticArgsHandler."""

    def test_initialize_resolves_static_argument_properties(self):
        """Test that initialize resolves static argument properties."""
        tool = _create_tool(
            "test_tool",
            {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
            },
        )
        handler = StaticArgsHandler()
        handler.initialize([tool], EmptyInput(), EmptyInput)

        call = _make_tool_call("test_tool")
        handler.apply_to_response([call])
        assert call["args"]["host"] == "api.example.com"

    def test_initialize_extracts_from_agent_input(self):
        """Test that initialize resolves AgentToolArgumentArgumentProperties from agent input."""

        class InputSchema(BaseModel):
            userId: str
            searchQuery: str

        tool = _create_tool(
            "test_tool",
            {
                "$['user_id']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="userId"
                ),
                "$['query']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="searchQuery"
                ),
            },
        )
        handler = StaticArgsHandler()
        state = InputSchema(userId="user123", searchQuery="test search")
        handler.initialize([tool], state, InputSchema)

        call = _make_tool_call("test_tool")
        handler.apply_to_response([call])
        assert call["args"]["user_id"] == "user123"
        assert call["args"]["query"] == "test search"

    def test_initialize_with_mixed_static_and_argument_properties(self):
        """Test initialize with both static and argument properties."""

        class InputSchema(BaseModel):
            userId: str

        tool = _create_tool(
            "test_tool",
            {
                "$['api_key']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="secret_key"
                ),
                "$['user_id']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="userId"
                ),
            },
        )
        handler = StaticArgsHandler()
        handler.initialize([tool], InputSchema(userId="user456"), InputSchema)

        call = _make_tool_call("test_tool")
        handler.apply_to_response([call])
        assert call["args"]["api_key"] == "secret_key"
        assert call["args"]["user_id"] == "user456"

    def test_initialize_skips_missing_argument_values(self):
        """Test that argument properties referencing missing agent input keys are skipped."""

        class InputSchema(BaseModel):
            existingArg: str
            missingArg: str = ""

        tool = _create_tool(
            "test_tool",
            {
                "$['existing_param']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="existingArg"
                ),
                "$['missing_param']": AgentToolArgumentArgumentProperties(
                    is_sensitive=False, argument_path="nonExistentField"
                ),
            },
        )
        handler = StaticArgsHandler()
        handler.initialize([tool], InputSchema(existingArg="exists"), InputSchema)

        call = _make_tool_call("test_tool")
        handler.apply_to_response([call])
        assert call["args"]["existing_param"] == "exists"
        assert "missing_param" not in call["args"]

    def test_apply_to_response_merges_with_existing_args(self):
        """Test that apply_to_response merges static args with existing tool call args."""
        tool = _create_tool(
            "test_tool",
            {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
            },
        )
        handler = StaticArgsHandler()
        handler.initialize([tool], EmptyInput(), EmptyInput)

        call = _make_tool_call("test_tool", {"port": 8080, "path": "/api"})
        handler.apply_to_response([call])
        assert call["args"] == {
            "host": "api.example.com",
            "port": 8080,
            "path": "/api",
        }

    def test_apply_to_response_ignores_unknown_tools(self):
        """Test that apply_to_response ignores tool calls for tools without static args."""
        tool = _create_tool(
            "test_tool",
            {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
            },
        )
        handler = StaticArgsHandler()
        handler.initialize([tool], EmptyInput(), EmptyInput)

        call = _make_tool_call("other_tool", {"query": "hello"})
        handler.apply_to_response([call])
        assert call["args"] == {"query": "hello"}

    def test_initialize_caches_results(self):
        """Test that initialize returns cached tools on subsequent calls."""
        tool = _create_tool(
            "test_tool",
            {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
            },
        )
        handler = StaticArgsHandler()
        tools_first = handler.initialize([tool], EmptyInput(), EmptyInput)
        tools_second = handler.initialize([tool], EmptyInput(), EmptyInput)
        assert tools_first is tools_second

    def test_initialize_returns_schema_modified_tools(self):
        """Test that initialize returns tools with schema modifications applied."""
        tool = _create_tool(
            "test_tool",
            {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
                "$['api_key']": AgentToolStaticArgumentProperties(
                    is_sensitive=True, value="secret-key-123"
                ),
            },
        )
        handler = StaticArgsHandler()
        processed_tools = handler.initialize([tool], EmptyInput(), EmptyInput)

        assert len(processed_tools) == 1
        modified_tool = processed_tools[0]
        assert modified_tool is not tool
        assert isinstance(modified_tool.args_schema, type) and issubclass(
            modified_tool.args_schema, BaseModel
        )
        assert isinstance(modified_tool, StructuredToolWithArgumentProperties)
        schema = modified_tool.args_schema.model_json_schema()
        assert "pre-configured" in schema["properties"]["api_key"]["description"]
        assert "api_key" not in schema["required"]
        host_def = schema["$defs"]["Host"]
        assert host_def["enum"] == ["api.example.com"]

    def test_initialize_returns_original_tool_when_no_static_args(self):
        """Test that tools without static argument properties are returned as-is."""
        tool = _create_tool("test_tool", {})
        handler = StaticArgsHandler()
        processed_tools = handler.initialize([tool], EmptyInput(), EmptyInput)

        assert len(processed_tools) == 1
        assert processed_tools[0] is tool

    def test_initialize_skips_nonexistent_schema_fields(self):
        """Test that static properties referencing nonexistent schema fields are skipped."""
        tool = _create_tool(
            "test_tool",
            {
                "$['nonexistent_field']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="test"
                ),
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
            },
        )
        handler = StaticArgsHandler()
        processed_tools = handler.initialize([tool], EmptyInput(), EmptyInput)

        modified_tool = processed_tools[0]
        assert isinstance(modified_tool.args_schema, type) and issubclass(
            modified_tool.args_schema, BaseModel
        )
        schema = modified_tool.args_schema.model_json_schema()
        host_def = schema["$defs"]["Host"]
        assert host_def["enum"] == ["api.example.com"]
        assert "nonexistent_field" not in schema["properties"]


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

    def test_apply_static_args_empty_array_does_not_skip_other_args(self):
        """Test empty array handling does not skip applying other static args."""
        static_args = {
            "$['files'][*]": {"id": "uuid-123"},
            "meta.source": "static-source",
        }
        kwargs: dict[str, Any] = {
            "files": [],
            "meta": {"source": "dynamic"},
        }

        result = apply_static_args(static_args, kwargs)

        assert result == {
            "files": [{"id": "uuid-123"}],
            "meta": {"source": "static-source"},
        }

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


class TestResolveStaticArgs:
    """Test cases for resolve_static_args function."""

    def test_resolve_static_args_with_argument_properties(self):
        """Test resolve_static_args with an object that has argument_properties."""

        class ResourceWithProps(ArgumentPropertiesMixin):
            argument_properties = {
                "$['host']": AgentToolStaticArgumentProperties(
                    is_sensitive=False, value="api.example.com"
                ),
            }

        result = resolve_static_args(ResourceWithProps(), {"unused": "input"})

        assert result == {"$['host']": "api.example.com"}

    def test_resolve_static_args_with_static_values_of_different_types(self):
        """Test resolve_static_args resolves string, integer, and object static values."""

        class ResourceWithProps(ArgumentPropertiesMixin):
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

        class ResourceWithProps(ArgumentPropertiesMixin):
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

        class ResourceWithProps(ArgumentPropertiesMixin):
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

        class ResourceWithProps(ArgumentPropertiesMixin):
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
