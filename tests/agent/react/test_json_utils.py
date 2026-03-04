from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel

from uipath_langchain.agent.react.json_utils import (
    extract_values_by_paths,
    get_json_paths_by_type,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model


class Target(BaseModel):
    id: str


class Nested(BaseModel):
    target: Target
    name: str


class WithList(BaseModel):
    items: list[Target]


class WithOptional(BaseModel):
    target: Optional[Target] = None


class WithNestedList(BaseModel):
    matrix: list[list[Target]]


class Mixed(BaseModel):
    direct: Target
    items: list[Target]
    nested: Nested
    plain: str


# -- get_json_paths_by_type: regular BaseModel ---------------------------------


class TestGetJsonPathsByType:
    def test_direct_field(self):
        paths = get_json_paths_by_type(Nested, "Target")
        assert paths == ["$.target"]

    def test_list_field(self):
        paths = get_json_paths_by_type(WithList, "Target")
        assert paths == ["$.items[*]"]

    def test_optional_field(self):
        paths = get_json_paths_by_type(WithOptional, "Target")
        assert paths == ["$.target"]

    def test_nested_list_of_lists(self):
        paths = get_json_paths_by_type(WithNestedList, "Target")
        assert paths == ["$.matrix[*][*]"]

    def test_mixed_fields(self):
        paths = get_json_paths_by_type(Mixed, "Target")
        assert set(paths) == {"$.direct", "$.items[*]", "$.nested.target"}

    def test_no_match(self):
        class Unrelated(BaseModel):
            name: str
            value: int

        paths = get_json_paths_by_type(Unrelated, "Target")
        assert paths == []

    def test_nested_model_in_list(self):
        class Outer(BaseModel):
            groups: list[Nested]

        paths = get_json_paths_by_type(Outer, "Target")
        assert paths == ["$.groups[*].target"]


# -- get_json_paths_by_type: RootModel -----------------------------------------


class TestGetJsonPathsByTypeRootModel:
    def test_root_model_single_object(self):
        Model = RootModel[Target]
        paths = get_json_paths_by_type(Model, "Target")
        assert paths == []

    def test_root_model_list_of_target(self):
        """Target is the leaf type itself — no nested fields of type Target within it."""
        Model = RootModel[list[Target]]
        paths = get_json_paths_by_type(Model, "Target")
        assert paths == []

    def test_root_model_list_of_nested(self):
        Model = RootModel[list[Nested]]
        paths = get_json_paths_by_type(Model, "Target")
        assert paths == ["$[*].target"]

    def test_root_model_list_of_list(self):
        Model = RootModel[list[list[Nested]]]
        paths = get_json_paths_by_type(Model, "Target")
        assert paths == ["$[*][*].target"]

    def test_root_model_primitive(self):
        Model = RootModel[str]
        paths = get_json_paths_by_type(Model, "Target")
        assert paths == []

    def test_root_model_list_of_primitives(self):
        Model = RootModel[list[str]]
        paths = get_json_paths_by_type(Model, "Target")
        assert paths == []

    def test_root_model_optional_list(self):
        Model = RootModel[Optional[list[Nested]]]
        paths = get_json_paths_by_type(Model, "Target")
        assert paths == ["$[*].target"]


# -- extract_values_by_paths ---------------------------------------------------


class TestExtractValuesByPaths:
    def test_extract_from_dict(self):
        data = {"target": {"id": "1"}, "name": "test"}
        values = extract_values_by_paths(data, ["$.target"])
        assert values == [{"id": "1"}]

    def test_extract_from_list(self):
        data = {"items": [{"id": "1"}, {"id": "2"}]}
        values = extract_values_by_paths(data, ["$.items[*]"])
        assert values == [{"id": "1"}, {"id": "2"}]

    def test_extract_from_pydantic_model(self):
        obj = Nested(target=Target(id="1"), name="test")
        values = extract_values_by_paths(obj, ["$.target"])
        assert values == [{"id": "1"}]

    def test_extract_multiple_paths(self):
        data = {
            "direct": {"id": "1"},
            "items": [{"id": "2"}, {"id": "3"}],
        }
        values = extract_values_by_paths(data, ["$.direct", "$.items[*]"])
        assert values == [{"id": "1"}, {"id": "2"}, {"id": "3"}]

    def test_extract_no_paths(self):
        values = extract_values_by_paths({"a": 1}, [])
        assert values == []

    def test_extract_path_not_found(self):
        values = extract_values_by_paths({"a": 1}, ["$.missing"])
        assert values == []


# -- aliased fields (renamed by create_model) ---------------------------------


class TestJsonPathsWithAliasedFields:
    """Verify that JSONPath extraction works correctly when fields have been
    renamed by create_model (underscore-prefixed or reserved names)."""

    def test_underscore_field_jsonpath_uses_alias(self):
        """JSONPath must use the original '_attachment' name, not the Python 'attachment'."""

        class Attachment(BaseModel):
            id: str

        class ModelWithAlias(BaseModel):
            model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

            attachment: Attachment = Field(
                alias="_attachment", serialization_alias="_attachment"
            )

        paths = get_json_paths_by_type(ModelWithAlias, "Attachment")
        assert paths == ["$._attachment"]

    def test_reserved_field_jsonpath_uses_alias(self):
        """JSONPath must use the original 'schema' alias, not the Python 'schema_'."""

        class Attachment(BaseModel):
            id: str

        class ModelWithReserved(BaseModel):
            model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

            schema_: Attachment = Field(alias="schema", serialization_alias="schema")

        paths = get_json_paths_by_type(ModelWithReserved, "Attachment")
        assert paths == ["$.schema"]

    def test_extract_values_from_dict_with_alias_keys(self):
        """extract_values_by_paths must find values using alias-keyed dicts."""

        class Attachment(BaseModel):
            id: str

        class ModelWithAlias(BaseModel):
            model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

            attachment: Attachment = Field(
                alias="_attachment", serialization_alias="_attachment"
            )

        paths = get_json_paths_by_type(ModelWithAlias, "Attachment")
        # Dict uses original/alias key names (as LLM or API would produce)
        data = {"_attachment": {"id": "abc-123"}}
        values = extract_values_by_paths(data, paths)
        assert values == [{"id": "abc-123"}]

    def test_extract_values_from_model_with_alias(self):
        """extract_values_by_paths on a BaseModel must dump with aliases."""

        class Attachment(BaseModel):
            id: str

        class ModelWithAlias(BaseModel):
            model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

            attachment: Attachment = Field(
                alias="_attachment", serialization_alias="_attachment"
            )

        paths = get_json_paths_by_type(ModelWithAlias, "Attachment")
        obj = ModelWithAlias.model_validate({"_attachment": {"id": "abc-123"}})
        values = extract_values_by_paths(obj, paths)
        assert values == [{"id": "abc-123"}]

    def test_create_model_with_underscore_attachment_field(self):
        """End-to-end: create_model + JSONPath for an underscore attachment field.

        Uses 'job-attachment' definition name (production format) which the library
        converts internally to namespace key '__Job_attachment'.
        """
        schema: dict[str, Any] = {
            "type": "object",
            "title": "Input",
            "properties": {
                "_file": {"$ref": "#/definitions/job-attachment"},
                "name": {"type": "string"},
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "full_name": {"type": "string"},
                    },
                }
            },
        }
        model = create_model(schema)

        # JSONPath should use the original "_file" name (the alias)
        paths = get_json_paths_by_type(model, "__Job_attachment")
        assert paths == ["$._file"]

        # Extract from a dict with original keys (as LLM would produce)
        data = {"_file": {"ID": "uuid-1", "full_name": "report.pdf"}, "name": "test"}
        values = extract_values_by_paths(data, paths)
        assert len(values) == 1
        assert values[0]["ID"] == "uuid-1"

    def test_create_model_with_reserved_attachment_field(self):
        """End-to-end: create_model + JSONPath for a reserved-name attachment field."""
        schema: dict[str, Any] = {
            "type": "object",
            "title": "Input",
            "properties": {
                "copy": {"$ref": "#/definitions/job-attachment"},
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "full_name": {"type": "string"},
                    },
                }
            },
        }
        model = create_model(schema)

        paths = get_json_paths_by_type(model, "__Job_attachment")
        assert paths == ["$.copy"]

        data = {"copy": {"ID": "uuid-2", "full_name": "backup.zip"}}
        values = extract_values_by_paths(data, paths)
        assert len(values) == 1
        assert values[0]["ID"] == "uuid-2"

    def test_create_model_attachment_list_with_underscore(self):
        """End-to-end: underscore attachment field inside a list."""
        schema: dict[str, Any] = {
            "type": "object",
            "title": "Input",
            "properties": {
                "_files": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                },
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "full_name": {"type": "string"},
                    },
                }
            },
        }
        model = create_model(schema)

        paths = get_json_paths_by_type(model, "__Job_attachment")
        assert paths == ["$._files[*]"]

        data = {
            "_files": [
                {"ID": "uuid-a", "full_name": "a.pdf"},
                {"ID": "uuid-b", "full_name": "b.pdf"},
            ]
        }
        values = extract_values_by_paths(data, paths)
        assert len(values) == 2
