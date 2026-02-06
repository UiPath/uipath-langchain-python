"""Tests for json_utils.py — JSONPath extraction from Pydantic models."""

from typing import Optional

from pydantic import BaseModel

from uipath_langchain.agent.react.json_utils import (
    _create_type_matcher,
    _is_pydantic_model,
    _unwrap_optional,
    extract_values_by_paths,
    get_json_paths_by_type,
)

# --- Test models ---


class Attachment(BaseModel):
    id: str
    filename: str


class NestedContainer(BaseModel):
    attachment: Attachment
    label: str


class ModelWithSimpleField(BaseModel):
    attachment: Attachment
    name: str


class ModelWithArrayField(BaseModel):
    attachments: list[Attachment]
    count: int


class ModelWithOptionalField(BaseModel):
    attachment: Optional[Attachment] = None
    name: str


class ModelWithNestedModel(BaseModel):
    container: NestedContainer
    title: str


class ModelWithArrayOfNested(BaseModel):
    containers: list[NestedContainer]


class ModelWithNoTargetType(BaseModel):
    name: str
    value: int


class ModelWithMixedFields(BaseModel):
    single: Attachment
    multiple: list[Attachment]
    label: str


# --- Tests for get_json_paths_by_type ---


class TestGetJsonPathsByType:
    """Tests for get_json_paths_by_type."""

    def test_simple_field(self) -> None:
        paths = get_json_paths_by_type(ModelWithSimpleField, "Attachment")
        assert paths == ["$.attachment"]

    def test_array_field(self) -> None:
        paths = get_json_paths_by_type(ModelWithArrayField, "Attachment")
        assert paths == ["$.attachments[*]"]

    def test_optional_field_unwrapped(self) -> None:
        paths = get_json_paths_by_type(ModelWithOptionalField, "Attachment")
        assert paths == ["$.attachment"]

    def test_nested_model_field(self) -> None:
        paths = get_json_paths_by_type(ModelWithNestedModel, "Attachment")
        assert paths == ["$.container.attachment"]

    def test_array_of_nested_models(self) -> None:
        paths = get_json_paths_by_type(ModelWithArrayOfNested, "Attachment")
        assert paths == ["$.containers[*].attachment"]

    def test_no_matching_type_returns_empty(self) -> None:
        paths = get_json_paths_by_type(ModelWithNoTargetType, "Attachment")
        assert paths == []

    def test_mixed_simple_and_array(self) -> None:
        paths = get_json_paths_by_type(ModelWithMixedFields, "Attachment")
        assert "$.single" in paths
        assert "$.multiple[*]" in paths
        assert len(paths) == 2


# --- Tests for extract_values_by_paths ---


class TestExtractValuesByPaths:
    """Tests for extract_values_by_paths."""

    def test_extract_from_dict_simple_path(self) -> None:
        obj = {"attachment": {"id": "123", "filename": "doc.pdf"}, "name": "test"}
        result = extract_values_by_paths(obj, ["$.attachment"])
        assert result == [{"id": "123", "filename": "doc.pdf"}]

    def test_extract_from_dict_array_path(self) -> None:
        obj = {
            "attachments": [
                {"id": "1", "filename": "a.pdf"},
                {"id": "2", "filename": "b.pdf"},
            ],
            "count": 2,
        }
        result = extract_values_by_paths(obj, ["$.attachments[*]"])
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_extract_from_basemodel(self) -> None:
        obj = ModelWithSimpleField(
            attachment=Attachment(id="456", filename="img.png"),
            name="test",
        )
        result = extract_values_by_paths(obj, ["$.attachment"])
        assert len(result) == 1
        assert result[0]["id"] == "456"

    def test_extract_multiple_paths(self) -> None:
        obj = {
            "single": {"id": "s1", "filename": "s.pdf"},
            "multiple": [
                {"id": "m1", "filename": "m1.pdf"},
                {"id": "m2", "filename": "m2.pdf"},
            ],
            "label": "test",
        }
        result = extract_values_by_paths(obj, ["$.single", "$.multiple[*]"])
        assert len(result) == 3

    def test_extract_no_match_returns_empty(self) -> None:
        obj = {"name": "test"}
        result = extract_values_by_paths(obj, ["$.nonexistent"])
        assert result == []

    def test_extract_empty_paths_returns_empty(self) -> None:
        obj = {"name": "test"}
        result = extract_values_by_paths(obj, [])
        assert result == []

    def test_extract_nested_path(self) -> None:
        obj = {
            "container": {
                "attachment": {"id": "nested", "filename": "n.pdf"},
                "label": "c",
            },
            "title": "t",
        }
        result = extract_values_by_paths(obj, ["$.container.attachment"])
        assert len(result) == 1
        assert result[0]["id"] == "nested"


# --- Tests for helper functions ---


class TestUnwrapOptional:
    """Tests for _unwrap_optional."""

    def test_unwraps_optional_type(self) -> None:
        result = _unwrap_optional(Optional[str])
        assert result is str

    def test_non_optional_unchanged(self) -> None:
        result = _unwrap_optional(str)
        assert result is str

    def test_unwraps_optional_basemodel(self) -> None:
        result = _unwrap_optional(Optional[Attachment])
        assert result is Attachment


class TestIsPydanticModel:
    """Tests for _is_pydantic_model."""

    def test_basemodel_returns_true(self) -> None:
        assert _is_pydantic_model(Attachment) is True

    def test_str_returns_false(self) -> None:
        assert _is_pydantic_model(str) is False

    def test_int_returns_false(self) -> None:
        assert _is_pydantic_model(int) is False

    def test_none_returns_false(self) -> None:
        assert _is_pydantic_model(None) is False

    def test_instance_returns_false(self) -> None:
        assert _is_pydantic_model(Attachment(id="1", filename="f")) is False


class TestCreateTypeMatcher:
    """Tests for _create_type_matcher."""

    def test_matches_by_name(self) -> None:
        matcher = _create_type_matcher("Attachment", None)
        assert matcher(Attachment) is True

    def test_matches_by_target_type(self) -> None:
        matcher = _create_type_matcher("Attachment", Attachment)
        assert matcher(Attachment) is True

    def test_no_match_returns_false(self) -> None:
        matcher = _create_type_matcher("Attachment", None)
        assert matcher(str) is False

    def test_string_annotation_match(self) -> None:
        matcher = _create_type_matcher("Attachment", None)
        assert matcher("Attachment") is True

    def test_string_annotation_no_match(self) -> None:
        matcher = _create_type_matcher("Attachment", None)
        assert matcher("OtherType") is False
