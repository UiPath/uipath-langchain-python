from typing import Annotated

import pytest
from pydantic import BaseModel

from uipath_langchain.agent.react.reducers import merge_objects


# Simple test models with annotations
def merge_strings(left: str, right: str) -> str:
    """Simple reducer that concatenates strings."""
    return f"{left or ''},{right or ''}"


def merge_lists(left: list[str], right: list[str]) -> list[str]:
    """Simple reducer that combines lists."""
    return (left or []) + (right or [])


def merge_dicts(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Simple reducer that merges dicts with right precedence."""
    return {**(left or {}), **(right or {})}


class SimpleModel(BaseModel):
    name: str = "default"
    value: int = 0


class ModelWithReducer(BaseModel):
    text: Annotated[str, merge_strings] = ""
    items: Annotated[list[str], merge_lists] = []
    mapping: Annotated[dict[str, str], merge_dicts] = {}
    count: int = 0


class TestMergeObjects:
    """Test merge_objects reducer function."""

    def test_empty_right_returns_left(self):
        """Should return left when right is empty."""
        left = SimpleModel()
        result = merge_objects(left, {})
        assert result is left

    def test_empty_left_returns_right(self):
        """Should return right when left is empty."""
        right = SimpleModel()
        result = merge_objects(None, right)
        assert result is right

    def test_left_not_basemodel_raises_error(self):
        """Should raise TypeError when left is not a BaseModel."""
        with pytest.raises(TypeError, match="Left object must be a Pydantic BaseModel"):
            merge_objects({"key": "value"}, {"some": "data"})

    def test_right_not_basemodel_or_dict_raises_error(self):
        """Should raise TypeError when right is not a BaseModel or dict."""
        left = SimpleModel()
        with pytest.raises(
            TypeError, match="Right object must be a Pydantic BaseModel or dict"
        ):
            merge_objects(left, "invalid")

    def test_simple_field_override_with_dict(self):
        """Should override simple fields when merging with dict."""
        left = SimpleModel(name="original", value=10)
        right = {"name": "updated", "value": 20}

        result = merge_objects(left, right)

        assert result.name == "updated"
        assert result.value == 20

    def test_simple_field_override_with_basemodel(self):
        """Should override simple fields when merging with BaseModel."""
        left = SimpleModel(name="original", value=10)
        right = SimpleModel(name="updated", value=20)

        result = merge_objects(left, right)

        assert result.name == "updated"
        assert result.value == 20

    def test_annotation_reducer_applied_with_dict(self):
        """Should apply annotation reducer when merging with dict."""
        left = ModelWithReducer(text="hello", items=["a", "b"], count=10)
        right = {"text": "world", "items": ["c", "d"], "count": 20}

        result = merge_objects(left, right)

        # text should be merged using string reducer
        assert result.text == "hello,world"
        # items should be merged using list reducer
        assert result.items == ["a", "b", "c", "d"]
        # count should be overridden (no reducer)
        assert result.count == 20

    def test_annotation_reducer_applied_with_basemodel(self):
        """Should apply annotation reducer when merging with BaseModel."""
        left = ModelWithReducer(text="hello", items=["a", "b"], count=10)
        right = ModelWithReducer(text="world", items=["c", "d"], count=20)

        result = merge_objects(left, right)

        # text should be merged using string reducer
        assert result.text == "hello,world"
        # items should be merged using list reducer
        assert result.items == ["a", "b", "c", "d"]
        # count should be overridden (no reducer)
        assert result.count == 20

    def test_dict_reducer_merges_correctly(self):
        """Should apply dict reducer correctly."""
        left = ModelWithReducer(mapping={"a": "1", "b": "2"})
        right = {"mapping": {"b": "3", "c": "4"}}

        result = merge_objects(left, right)

        # mapping should be merged with right precedence
        assert result.mapping == {"a": "1", "b": "3", "c": "4"}

    def test_mixed_fields_with_and_without_reducers(self):
        """Should handle fields with reducers and simple override fields in same merge."""
        left = ModelWithReducer(
            text="hello", items=["a", "b"], mapping={"x": "1"}, count=10
        )
        right = ModelWithReducer(
            text="world", items=["c", "d"], mapping={"y": "2"}, count=20
        )

        result = merge_objects(left, right)

        # Fields with reducers should be merged
        assert result.text == "hello,world"
        assert result.items == ["a", "b", "c", "d"]
        assert result.mapping == {"x": "1", "y": "2"}
        # Field without reducer should be overridden
        assert result.count == 20

    def test_field_not_present_in_right_keeps_left_value(self):
        """Should keep left value when field is not present in right."""
        left = ModelWithReducer(text="hello", items=["a", "b"], count=10)
        right: dict[str, object] = {}  # Empty dict

        result = merge_objects(left, right)

        # All left values should be preserved
        assert result.text == "hello"
        assert result.items == ["a", "b"]
        assert result.count == 10

    def test_custom_model_with_annotation_reducer(self):
        """Should work with custom models that have annotation reducers."""
        left = ModelWithReducer(items=["a", "b"], text="left")
        right = ModelWithReducer(items=["c", "d"], text="right")

        result = merge_objects(left, right)

        # items should be merged using list reducer
        assert result.items == ["a", "b", "c", "d"]
        # text should be merged using string reducer
        assert result.text == "left,right"

    def test_empty_values_handled_correctly(self):
        """Should handle empty values correctly in reducer application."""
        left = ModelWithReducer()  # All defaults
        right = ModelWithReducer(text="hello", items=["a", "b"])

        result = merge_objects(left, right)

        # Should handle empty left values correctly
        assert result.text == ",hello"  # Empty string + "hello"
        assert result.items == ["a", "b"]  # Empty list + ["a", "b"]

    def test_invalid_field_names_ignored(self):
        """Should ignore fields in right dict that don't exist in left model."""
        left = SimpleModel(name="test", value=10)
        right = {"name": "updated", "invalid_field": "should be ignored"}

        result = merge_objects(left, right)

        # Valid field should be set
        assert result.name == "updated"
        # Invalid field should not exist
        assert not hasattr(result, "invalid_field")
