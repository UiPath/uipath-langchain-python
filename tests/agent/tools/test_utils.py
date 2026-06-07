"""Tests for tools/utils.py module."""

from pydantic import BaseModel

from uipath_langchain.agent.tools.utils import (
    sanitize_dict_for_serialization,
    sanitize_tool_name,
)


class MockEnumLike:
    """Mock class that simulates an enum-like object with a value attribute."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"<Send as.KEY_0: '{self.value}'>"


class MockPydanticModel(BaseModel):
    """Mock Pydantic model for testing."""

    name: str
    value: int


class TestSanitizeToolName:
    """Test cases for sanitize_tool_name function."""

    def test_basic_sanitization(self):
        """Test basic tool name sanitization."""
        result = sanitize_tool_name("test_tool")
        assert result == "test_tool"

    def test_whitespace_replacement(self):
        """Test that whitespaces are replaced with underscores."""
        result = sanitize_tool_name("test tool name")
        assert result == "test_tool_name"

    def test_special_character_removal(self):
        """Test that special characters are removed."""
        result = sanitize_tool_name("test@tool#name!")
        assert result == "testtoolname"

    def test_length_truncation(self):
        """Test that long names are truncated to 64 characters."""
        long_name = "a" * 100
        result = sanitize_tool_name(long_name)
        assert len(result) == 64
        assert result == "a" * 64


class TestSanitizeDictForSerialization:
    """Test cases for sanitize_dict_for_serialization function."""

    def test_regular_values_unchanged(self):
        """Test that regular values pass through unchanged."""
        input_dict = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        result = sanitize_dict_for_serialization(input_dict)
        assert result == input_dict

    def test_pydantic_model_conversion(self):
        """Test that Pydantic models are converted to dicts."""
        model = MockPydanticModel(name="test", value=42)
        input_dict = {"model": model}

        result = sanitize_dict_for_serialization(input_dict)

        assert result["model"] == {"name": "test", "value": 42}

    def test_enum_like_object_conversion(self):
        """Test that enum-like objects with value attribute are converted."""
        enum_obj = MockEnumLike("bot")
        input_dict = {"enum": enum_obj}

        result = sanitize_dict_for_serialization(input_dict)

        assert result["enum"] == "bot"

    def test_list_with_pydantic_models(self):
        """Test that lists containing Pydantic models are handled."""
        models = [
            MockPydanticModel(name="first", value=1),
            MockPydanticModel(name="second", value=2),
        ]
        input_dict = {"models": models}

        result = sanitize_dict_for_serialization(input_dict)

        expected = [
            {"name": "first", "value": 1},
            {"name": "second", "value": 2},
        ]
        assert result["models"] == expected

    def test_list_with_enum_like_objects(self):
        """Test that lists with enum-like objects are handled."""
        enum_objects = [MockEnumLike("bot"), MockEnumLike("user")]
        input_dict = {"enums": enum_objects}

        result = sanitize_dict_for_serialization(input_dict)

        assert result["enums"] == ["bot", "user"]

    def test_object_without_value_attribute(self):
        """Test that objects without value attribute pass through unchanged."""

        class NoValueObject:
            def __init__(self):
                self.other_attr = "test"

        obj = NoValueObject()
        input_dict = {"obj": obj}

        result = sanitize_dict_for_serialization(input_dict)

        assert result["obj"] is obj
