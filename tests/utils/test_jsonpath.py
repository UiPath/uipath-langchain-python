import pytest

from uipath_langchain._utils._jsonpath import (
    parse_jsonpath_segments,
)


class TestParseJsonpathSegments:
    """Test cases for parse_jsonpath_segments function."""

    @pytest.mark.parametrize(
        "json_path,expected",
        [
            ("$['host']", ["host"]),
            ("$['config']['host']", ["config", "host"]),
            ("$['a']['b']['c']", ["a", "b", "c"]),
            ('$["host"]', ["host"]),
            ("$['tags'][*]['title']", ["tags", "*", "title"]),
        ],
    )
    def test_parse_jsonpath_segments(self, json_path: str, expected: list[str]):
        result = parse_jsonpath_segments(json_path)
        assert result == expected
