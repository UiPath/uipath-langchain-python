"""Tests for the JSON schema editing utilities."""

from typing import Any, Optional

import pytest
from jsonpath_ng import parse  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from uipath_langchain.agent.tools.schema_editing import (
    SchemaModificationError,
    apply_static_value_to_schema,
)


def _get_parent_path(field_path: str) -> str:
    """Compute the parent's schema path. Works only for dot-separated paths."""
    parts = field_path.split(".")
    parts = parts[:-1]
    if parts and parts[-1] == "properties":
        parts = parts[:-1]
    return ".".join(parts)


def _assert_sensitive_field(schema: dict[str, Any], field_path: str) -> None:
    """Assert that a field has been correctly marked as sensitive."""
    field_name = field_path.split(".")[-1]
    parent_path = _get_parent_path(field_path)

    # Check description
    description_matches = parse(f"{field_path}.description").find(schema)
    assert description_matches, f"Description not found for path {field_path}"
    description = description_matches[0].value
    assert "This argument is pre-configured" in description, (
        f"Expected '{field_name}' description to contain 'This argument is pre-configured', "
        f"but got: {description}"
    )

    # Check that field is not required
    required_matches = parse(f"{parent_path}.required").find(schema)
    required_fields = required_matches[0].value if required_matches else []
    assert field_name not in required_fields, (
        f"Field '{field_name}' should not be required for sensitive fields, "
        f"but was found in required list: {required_fields}"
    )


def _assert_non_sensitive_field(
    schema: dict[str, Any], field_path: str, expected_value: Any
) -> None:
    """Assert that a field has been correctly marked as non-sensitive with enum."""
    field_name = field_path.split(".")[-1]

    # Check for enum value
    enum_path = f"{field_path}.enum"
    enum_matches = parse(enum_path).find(schema)
    assert enum_matches, f"Enum not found at path: {enum_path}"
    enum_value = enum_matches[0].value

    assert expected_value in enum_value, (
        f"Expected '{field_name}' enum to equal {expected_value}, but got: {enum_value}"
    )


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    host: str
    port: int = Field(default=8080)
    api_key: str


class NestedInput(BaseModel):
    """Nested input model for testing."""

    class ServerConfig(BaseModel):
        host: str
        port: int

    user_id: str
    config: ServerConfig


class TestApplyStaticValueToSchema:
    """Test that the schema is modified correctly when a static value is applied."""

    def test_sensitive_param_makes_optional(self):
        """Test that sensitive params get placeholder description and are removed from required."""
        schema = SimpleInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['api_key']",
            value="secret-key-123",
            is_sensitive=True,
        )

        _assert_sensitive_field(schema, "$.properties.api_key")

    def test_non_sensitive_primitive_gets_enum(self):
        """Test that non-sensitive primitive params get enum values."""
        schema = SimpleInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['host']",
            value="api.example.com",
            is_sensitive=False,
        )

        _assert_non_sensitive_field(schema, "$.properties.host", "api.example.com")

    @pytest.mark.parametrize(
        "value",
        [
            "production",
            42,
            True,
            3.14,
        ],
    )
    def test_various_primitive_types_get_enum(self, value):
        """Test that various primitive types get proper enum values."""
        schema = SimpleInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['host']",
            value=value,
            is_sensitive=False,
        )

        _assert_non_sensitive_field(schema, "$.properties.host", value)

    def test_multiple_nested_modifications(self):
        """Test multiple nested modifications on the same schema."""

        schema = NestedInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['config']['host']",
            value="prod-server.com",
            is_sensitive=False,
        )
        apply_static_value_to_schema(
            schema,
            json_path="$['config']['port']",
            value=443,
            is_sensitive=False,
        )

        _assert_non_sensitive_field(
            schema, "$.properties.config.properties.host", "prod-server.com"
        )
        _assert_non_sensitive_field(schema, "$.properties.config.properties.port", 443)


class ArrayInput(BaseModel):
    """Input model with array for testing."""

    tags: list[str]
    name: str


class TestApplyArrayStaticValueToSchema:
    """Test cases for array schema modifications."""

    def test_array_converted_to_json_string_enum(self):
        """Test that array values are converted to JSON-serialized string enum."""
        schema = ArrayInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['tags']",
            value=["tag1", "tag2", "tag3"],
            is_sensitive=False,
        )

        # Array should be converted to string type with JSON-serialized enum
        assert schema["properties"]["tags"]["type"] == "string"
        _assert_non_sensitive_field(
            schema, "$.properties.tags", '["tag1", "tag2", "tag3"]'
        )


class TestApplyNestedStaticValueToSchema:
    """Test cases for nested object required field modifications."""

    def test_nested_sensitive_field_removed_from_nested_required(self):
        """Test that nested sensitive fields are removed from the nested object's required array."""
        schema = NestedInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['config']['host']",
            value="secret-host.com",
            is_sensitive=True,
        )

        _assert_sensitive_field(schema, "$.properties.config.properties.host")
        assert "config" in schema.get("required", [])
        assert "port" in schema["properties"]["config"].get("required", [])

    def test_top_level_sensitive_field_removed_from_top_required(self):
        """Test that top-level sensitive fields are removed from the top-level required array."""
        schema = NestedInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['user_id']",
            value="secret-user-123",
            is_sensitive=True,
        )

        _assert_sensitive_field(schema, "$.properties.user_id")
        assert "config" in schema.get("required", [])


class TestApplyStaticValueToSchemaWithRefs:
    """Test cases for applying static values to schemas with $refs."""

    def test_modifying_one_ref_doesnt_affect_others(self):
        """Test that modifying a field in one object doesn't affect other objects sharing the same $def."""

        class SharedConfig(BaseModel):
            host: str
            port: int

        class MultipleConfigsInput(BaseModel):
            config1: SharedConfig
            config2: SharedConfig

        schema = MultipleConfigsInput.model_json_schema()

        # Modify only config1.host
        apply_static_value_to_schema(
            schema,
            json_path="$['config1']['host']",
            value="static-host.com",
            is_sensitive=False,
        )

        # config1 should be inlined
        _assert_non_sensitive_field(
            schema, "$.properties.config1.properties.host", "static-host.com"
        )
        config1_schema = schema["properties"]["config1"]
        assert "port" in config1_schema["properties"]
        assert "enum" not in config1_schema["properties"]["port"]

        # config2 should be unchanged
        config2_schema = schema["properties"]["config2"]
        assert "$ref" in config2_schema
        assert config2_schema["$ref"] == "#/$defs/SharedConfig"

    def test_nested_ref_in_defs(self):
        """Test that modifications work when a $def contains a $ref to another $def."""

        class Address(BaseModel):
            street: str
            city: str

        class ServerConfig(BaseModel):
            host: str
            address: Address

        class ServerInput(BaseModel):
            server: ServerConfig

        schema = ServerInput.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['server']['address']['street']",
            value="secret-street",
            is_sensitive=True,
        )

        _assert_sensitive_field(
            schema, "$.properties.server.properties.address.properties.street"
        )
        # city should still be required
        address_schema = schema["properties"]["server"]["properties"]["address"]
        assert "city" in address_schema.get("required", [])


class FilesModel(BaseModel):
    class File(BaseModel):
        id: str

    files: list[File]


class TestArrayItemsSchemaModification:
    """Test cases for modifying fields inside array items schemas."""

    def test_modify_array_items_object(self):
        schema = FilesModel.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['files'][*]",
            value={"id": "123"},
            is_sensitive=False,
        )

        item_schema = schema["properties"]["files"]["items"]
        assert item_schema["properties"]["id"]["enum"] == ["123"]

    def test_modify_sensitive_array_items_affects_array_property(self):
        schema = FilesModel.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['files'][*]",
            value={"id": "secret-id"},
            is_sensitive=True,
        )

        # The affected field should be files, not files.items
        _assert_sensitive_field(schema, "$.properties.files")

    def test_modify_field_in_array_items_primitive(self):
        """Test modifying a primitive field inside array items."""

        schema = FilesModel.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['files'][*]['id']",
            value="static-id",
            is_sensitive=False,
        )

        _assert_non_sensitive_field(
            schema, "$.properties.files.*.properties.id", "static-id"
        )

    def test_modify_sensitive_field_in_array_items(self):
        """Test modifying a sensitive field inside array items."""

        schema = FilesModel.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['files'][*]['id']",
            value="secret-name",
            is_sensitive=True,
        )

        _assert_sensitive_field(schema, "$.properties.files.items.properties.id")

    def test_modify_array_field_inside_array_items(self):
        """Test modifying an array field that's inside array items."""

        class Task(BaseModel):
            title: str
            tags: list[str]

        class Project(BaseModel):
            tasks: list[Task]

        schema = Project.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['tasks'][*]['tags']",
            value=["urgent", "backend"],
            is_sensitive=False,
        )

        _assert_non_sensitive_field(
            schema, "$.properties.tasks.items.properties.tags", '["urgent", "backend"]'
        )
        items_schema = schema["properties"]["tasks"]["items"]
        assert items_schema["properties"]["tags"]["type"] == "string"


class TestInvalidSchemaPath:
    """Test cases for error handling with invalid schema paths."""

    def test_invalid_field_name_raises_error(self):
        """Test that accessing a non-existent field raises SchemaModificationError."""

        class SimpleModel(BaseModel):
            name: str
            value: int

        schema = SimpleModel.model_json_schema()

        with pytest.raises(
            SchemaModificationError, match="Invalid schema path.*nonexistent"
        ):
            apply_static_value_to_schema(
                schema,
                json_path="$['nonexistent']",
                value="test",
                is_sensitive=False,
            )

    def test_empty_json_path_raises_error(self):
        """Test that an empty JSON path raises InvalidSchemaPathError."""

        class Model(BaseModel):
            name: str

        schema = Model.model_json_schema()

        with pytest.raises(SchemaModificationError, match="Empty JSON path"):
            apply_static_value_to_schema(
                schema,
                json_path="$",
                value="test",
                is_sensitive=False,
            )

    def test_invalid_array_navigation_raises_error(self):
        """Test that trying to navigate an array incorrectly raises InvalidSchemaPathError."""

        class Model(BaseModel):
            tags: list[str]

        schema = Model.model_json_schema()

        # Trying to access a field on array without using 'items' keyword
        with pytest.raises(SchemaModificationError):
            apply_static_value_to_schema(
                schema,
                json_path="$['tags']['name']",
                value="test",
                is_sensitive=False,
            )


class TestApplyStaticValueToSchemaWithAnyOfUnion:
    """Test cases for applying static values to schemas with anyOf union."""

    def test_apply_static_value_to_schema_with_root_optional_field(self):
        class TestModel(BaseModel):
            name: Optional[str]

        schema = TestModel.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['name']",
            value="test",
            is_sensitive=False,
        )

        union_list = schema["properties"]["name"]["anyOf"]
        assert len(union_list) == 2
        assert union_list[0]["type"] == "string"
        assert union_list[0]["enum"] == ["test"]
        assert union_list[1]["type"] == "null"

    def test_apply_static_values_to_schema_with_nested_optional_fields(self):
        class TestModel(BaseModel):
            class Nested(BaseModel):
                class Name(BaseModel):
                    final: Optional[str]
                    secret: Optional[int]

                name: Optional[Name]

            nested: Optional[Nested]

        schema = TestModel.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['nested']['name']['final']",
            value="test",
            is_sensitive=False,
        )

        apply_static_value_to_schema(
            schema,
            json_path="$['nested']['name']['secret']",
            value=123,
            is_sensitive=True,
        )

        nested_union_list = schema["properties"]["nested"]["anyOf"]
        assert len(nested_union_list) == 2
        assert nested_union_list[1]["type"] == "null"

        nested_schema = nested_union_list[0]
        assert nested_schema["type"] == "object"
        assert "anyOf" in nested_schema["properties"]["name"]

        name_union_list = nested_schema["properties"]["name"]["anyOf"]
        assert len(name_union_list) == 2
        assert name_union_list[1]["type"] == "null"

        name_schema = name_union_list[0]
        assert name_schema["type"] == "object"
        assert "anyOf" in name_schema["properties"]["final"]

        final_union_list = name_schema["properties"]["final"]["anyOf"]
        assert len(final_union_list) == 2
        assert final_union_list[1]["type"] == "null"

        final_schema = final_union_list[0]
        assert final_schema["type"] == "string"
        assert final_schema["enum"] == ["test"]

        secret_union_list = name_schema["properties"]["secret"]["anyOf"]
        assert len(secret_union_list) == 2
        assert secret_union_list[1]["type"] == "null"

        secret_schema = secret_union_list[0]
        assert secret_schema["type"] == "integer"
        assert "pre-configured" in secret_schema["description"]
        assert "secret" not in name_schema["required"]

    def test_apply_static_values_to_schema_with_optional_array_item(self):
        class TestModel(BaseModel):
            class Item(BaseModel):
                value: str

            array: list[Optional[Item]]

        schema = TestModel.model_json_schema()

        apply_static_value_to_schema(
            schema,
            json_path="$['array'][*]['value']",
            value="static_value",
            is_sensitive=False,
        )

        items_union_list = schema["properties"]["array"]["items"]["anyOf"]
        assert len(items_union_list) == 2
        assert items_union_list[1]["type"] == "null"
        items_schema = items_union_list[0]
        assert items_schema["type"] == "object"
        assert items_schema["properties"]["value"]["type"] == "string"
        assert items_schema["properties"]["value"]["enum"] == ["static_value"]
