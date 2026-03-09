from typing import Any, Optional

from pydantic import BaseModel, RootModel

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


# -- get_json_paths_by_type: dynamic models from create_model ------------------


class TestGetJsonPathsByTypeDynamic:
    """Exercise _get_target_type pseudo-module lookup with dynamic models."""

    def test_direct_ref(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "attachment": {"$ref": "#/definitions/job-attachment"},
                "name": {"type": "string"},
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                    },
                }
            },
        }
        model = create_model(schema)
        paths = get_json_paths_by_type(model, "__Job_attachment")
        assert paths == ["$.attachment"]

    def test_array_ref(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {"ID": {"type": "string"}},
                }
            },
        }
        model = create_model(schema)
        paths = get_json_paths_by_type(model, "__Job_attachment")
        assert paths == ["$.items[*]"]

    def test_nested_ref_only_in_defs(self) -> None:
        """Type referenced only by another $def, not directly by root."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "order": {"$ref": "#/$defs/Order"},
            },
            "$defs": {
                "Order": {
                    "type": "object",
                    "properties": {
                        "item": {"$ref": "#/$defs/Item"},
                        "quantity": {"type": "integer"},
                    },
                },
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }
        model = create_model(schema)
        paths = get_json_paths_by_type(model, "__Item")
        assert paths == ["$.order.item"]


class TestJsonPathsWithAliasedFields:
    """Verify JSONPath extraction works with renamed fields from create_model."""

    def test_underscore_field_jsonpath(self) -> None:
        """JSONPath must use original '_file' name, not Python 'file'."""
        schema: dict[str, Any] = {
            "type": "object",
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
        paths = get_json_paths_by_type(model, "__Job_attachment")
        assert paths == ["$._file"]

    def test_reserved_field_jsonpath(self) -> None:
        """JSONPath must use original 'copy' name."""
        schema: dict[str, Any] = {
            "type": "object",
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

    def test_underscore_field_extract_from_dict(self) -> None:
        """extract_values_by_paths must find values using alias-keyed dicts."""
        schema: dict[str, Any] = {
            "type": "object",
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
        paths = get_json_paths_by_type(model, "__Job_attachment")
        data = {"_file": {"ID": "uuid-1", "full_name": "report.pdf"}, "name": "test"}
        values = extract_values_by_paths(data, paths)
        assert len(values) == 1
        assert values[0]["ID"] == "uuid-1"

    def test_underscore_field_list_jsonpath(self) -> None:
        """Underscore attachment field inside a list."""
        schema: dict[str, Any] = {
            "type": "object",
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
