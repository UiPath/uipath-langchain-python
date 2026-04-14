"""Tests for Data Fabric subgraph — structured error feedback and convergence."""

import json

from uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph import (
    _classify_error_hint,
    _format_tool_result,
)


class TestClassifyErrorHint:
    def test_missing_limit(self):
        hint = _classify_error_hint("Queries without WHERE must include a LIMIT clause")
        assert "LIMIT" in hint

    def test_select_star(self):
        hint = _classify_error_hint("SELECT * is not allowed")
        assert "SELECT *" in hint or "explicit" in hint.lower()

    def test_count_star(self):
        hint = _classify_error_hint("COUNT(*) not allowed")
        assert "COUNT(column_name)" in hint

    def test_unknown_column(self):
        hint = _classify_error_hint("unknown column 'foo' in field list")
        assert "column" in hint.lower()

    def test_no_such_table(self):
        hint = _classify_error_hint("no such table: Foobar")
        assert "table" in hint.lower()

    def test_right_join(self):
        hint = _classify_error_hint("RIGHT JOIN is not supported")
        assert "LEFT JOIN" in hint

    def test_syntax_error(self):
        hint = _classify_error_hint("syntax error near 'FORM'")
        assert "syntax" in hint.lower()

    def test_generic_fallback(self):
        hint = _classify_error_hint("some totally unknown error XYZ")
        assert hint  # should still return something


class TestFormatToolResult:
    def test_success_result(self):
        result = {
            "records": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "total_count": 2,
            "sql_query": "SELECT id, name FROM Customer LIMIT 2",
        }
        formatted = _format_tool_result(result)
        parsed = json.loads(formatted)
        assert parsed["status"] == "success"
        assert parsed["row_count"] == 2
        assert len(parsed["sample_rows"]) == 2
        assert parsed["sql_query"] == result["sql_query"]

    def test_error_result(self):
        result = {
            "records": [],
            "total_count": 0,
            "error": "unknown column 'foo'",
            "sql_query": "SELECT foo FROM Customer LIMIT 1",
        }
        formatted = _format_tool_result(result)
        parsed = json.loads(formatted)
        assert parsed["status"] == "error"
        assert "unknown column" in parsed["error_message"]
        assert "hint" in parsed

    def test_zero_results_hint(self):
        result = {
            "records": [],
            "total_count": 0,
            "sql_query": "SELECT id FROM Customer WHERE name = 'nonexistent' LIMIT 1",
        }
        formatted = _format_tool_result(result)
        parsed = json.loads(formatted)
        assert parsed["status"] == "success"
        assert parsed["row_count"] == 0
        assert "hint" in parsed
        assert "0 results" in parsed["hint"]

    def test_sample_rows_limited_to_five(self):
        result = {
            "records": [{"id": i} for i in range(20)],
            "total_count": 20,
            "sql_query": "SELECT id FROM Customer LIMIT 20",
        }
        formatted = _format_tool_result(result)
        parsed = json.loads(formatted)
        assert len(parsed["sample_rows"]) == 5

    def test_result_is_valid_json(self):
        result = {
            "records": [{"name": "O'Reilly"}],
            "total_count": 1,
            "sql_query": "SELECT name FROM Customer WHERE name = 'O''Reilly' LIMIT 1",
        }
        formatted = _format_tool_result(result)
        # Should not raise
        parsed = json.loads(formatted)
        assert parsed["status"] == "success"
