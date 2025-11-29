"""Tests for agent tools utility functions."""

import pytest

from uipath_langchain.agent.tools.utils import sanitize_tool_name


class TestSanitizeToolName:
    """Test cases for sanitize_tool_name function."""

    def test_simple_alphanumeric(self):
        """Should preserve simple alphanumeric names."""
        assert sanitize_tool_name("my_tool") == "my_tool"
        assert sanitize_tool_name("MyTool123") == "MyTool123"

    def test_preserves_underscores(self):
        """Should preserve underscores in tool names."""
        assert sanitize_tool_name("my_tool_name") == "my_tool_name"

    def test_preserves_hyphens(self):
        """Should preserve hyphens in tool names."""
        assert sanitize_tool_name("my-tool-name") == "my-tool-name"

    def test_removes_special_characters(self):
        """Should remove special characters not allowed."""
        assert sanitize_tool_name("my@tool!name") == "mytoolname"
        assert sanitize_tool_name("tool#$%") == "tool"
        assert sanitize_tool_name("tool(1)") == "tool1"

    def test_converts_spaces_to_underscores(self):
        """Should convert whitespace to underscores."""
        assert sanitize_tool_name("my tool name") == "my_tool_name"
        assert sanitize_tool_name("my  tool") == "my_tool"

    def test_handles_multiple_spaces(self):
        """Should handle multiple consecutive spaces by joining with single underscore."""
        assert sanitize_tool_name("tool   name") == "tool_name"

    def test_truncates_long_names(self):
        """Should truncate names longer than 64 characters."""
        long_name = "a" * 100
        result = sanitize_tool_name(long_name)
        assert len(result) == 64
        assert result == "a" * 64

    def test_empty_string(self):
        """Should handle empty string input."""
        assert sanitize_tool_name("") == ""

    def test_only_special_characters(self):
        """Should return empty string when only special characters."""
        assert sanitize_tool_name("@#$%^&*()") == ""

    def test_unicode_characters(self):
        """Should remove unicode/non-ASCII characters."""
        assert sanitize_tool_name("tool_name_") == "tool_name_"
        assert sanitize_tool_name("cafe") == "cafe"

    def test_leading_trailing_spaces(self):
        """Should handle leading and trailing spaces."""
        assert sanitize_tool_name("  tool_name  ") == "tool_name"

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("Simple", "Simple"),
            ("with spaces", "with_spaces"),
            ("with-hyphens", "with-hyphens"),
            ("with_underscores", "with_underscores"),
            ("MixedCase123", "MixedCase123"),
            ("special!@#chars", "specialchars"),
            ("a" * 70, "a" * 64),
        ],
    )
    def test_various_inputs(self, input_name, expected):
        """Should handle various input patterns correctly."""
        assert sanitize_tool_name(input_name) == expected
