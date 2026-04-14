"""Tests for the DSPy optimizer components (metrics and export)."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from uipath_langchain.agent.tools.datafabric_tool.optimizer.export import (
    export_optimized_prompts,
    load_optimized_prompts,
)
from uipath_langchain.agent.tools.datafabric_tool.optimizer.metrics import (
    _normalize_sql,
    sql_match_metric,
)


class TestNormalizeSql:
    def test_basic_normalization(self):
        assert _normalize_sql("  SELECT  id  FROM  T  ; ") == "select id from t"

    def test_comma_spacing(self):
        assert _normalize_sql("SELECT a,b,c FROM T") == "select a, b, c from t"

    def test_parenthesis_spacing(self):
        result = _normalize_sql("COUNT( id )")
        assert result == "count(id)"


class TestSqlMatchMetric:
    def _ex(self, sql: str) -> SimpleNamespace:
        return SimpleNamespace(sql=sql)

    def test_exact_match(self):
        ex = self._ex("SELECT id FROM Customer LIMIT 100")
        pred = self._ex("SELECT id FROM Customer LIMIT 100")
        assert sql_match_metric(ex, pred) == 1.0

    def test_match_after_normalization(self):
        ex = self._ex("SELECT id FROM Customer LIMIT 100;")
        pred = self._ex("  select  id  from  customer  limit  100  ")
        assert sql_match_metric(ex, pred) == 1.0

    def test_no_match(self):
        ex = self._ex("SELECT id FROM Customer LIMIT 100")
        pred = self._ex("SELECT name FROM Orders WHERE status = 'active'")
        assert sql_match_metric(ex, pred) == 0.0

    def test_empty_prediction(self):
        ex = self._ex("SELECT id FROM Customer LIMIT 100")
        pred = self._ex("")
        assert sql_match_metric(ex, pred) == 0.0


class TestExportOptimizedPrompts:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "optimized.json"
            written_path = export_optimized_prompts(
                optimized_instruction="Be precise",
                few_shot_examples=[
                    {"question": "How many?", "sql": "SELECT COUNT(id) FROM T LIMIT 1"}
                ],
                use_reasoning=True,
                optimizer_name="miprov2",
                val_accuracy=0.91,
                output_path=path,
            )
            assert written_path == path
            assert path.exists()

            loaded = load_optimized_prompts(path)
            assert loaded is not None
            assert loaded["optimized_instruction"] == "Be precise"
            assert len(loaded["few_shot_examples"]) == 1
            assert loaded["use_reasoning"] is True
            assert loaded["optimization_metadata"]["optimizer"] == "miprov2"
            assert loaded["optimization_metadata"]["val_accuracy"] == 0.91

    def test_load_missing_file(self):
        result = load_optimized_prompts(Path("/nonexistent/optimized.json"))
        assert result is None

    def test_export_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "optimized.json"
            export_optimized_prompts(
                optimized_instruction="test",
                few_shot_examples=[],
                use_reasoning=False,
                optimizer_name="bootstrap",
                val_accuracy=0.5,
                output_path=path,
            )
            assert path.exists()

    def test_export_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "optimized.json"
            export_optimized_prompts(
                optimized_instruction="test",
                few_shot_examples=[],
                use_reasoning=False,
                optimizer_name="bootstrap",
                val_accuracy=0.85,
                output_path=path,
            )
            data = json.loads(path.read_text())
            assert "optimized_instruction" in data
            assert "few_shot_examples" in data
            assert "use_reasoning" in data
            assert "optimization_metadata" in data
            assert "timestamp" in data["optimization_metadata"]
