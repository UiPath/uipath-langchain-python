"""DSPy signatures for Text2SQL generation.

Two variants are provided:
- ``Text2SQL``  — direct question->SQL generation.
- ``Text2SQLWithReasoning``  — adds an intermediate reasoning step
  (CoT schema linking) before SQL output.

The optimizer compares both and picks the higher-scoring variant.

Note: ``dspy`` is an optional dependency.  The classes are defined inside
a factory function so this module can be imported without dspy installed
(required by the circular-import test suite).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def get_signatures() -> tuple[Any, Any]:
    """Return (Text2SQL, Text2SQLWithReasoning) DSPy Signature classes.

    Raises:
        ImportError: If ``dspy`` is not installed.
    """
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "dspy is required for the optimizer.  "
            "Install with:  uv sync --extra optimize"
        ) from exc

    class Text2SQL(dspy.Signature):
        """Convert a natural language question into a SQL query given entity schemas."""

        question: str = dspy.InputField(desc="Natural language question about the data")
        schema: str = dspy.InputField(
            desc="Entity schemas with field names, types, keys, descriptions"
        )
        sql: str = dspy.OutputField(
            desc="Valid SQL SELECT query using exact table/column names from the schema"
        )

    class Text2SQLWithReasoning(dspy.Signature):
        """Convert a natural language question into SQL, reasoning about schema first."""

        question: str = dspy.InputField(desc="Natural language question about the data")
        schema: str = dspy.InputField(
            desc="Entity schemas with field names, types, keys, descriptions"
        )
        reasoning: str = dspy.OutputField(
            desc="Which tables, columns, filters, and aggregations are needed"
        )
        sql: str = dspy.OutputField(
            desc="Valid SQL SELECT query using exact table/column names from the schema"
        )

    return Text2SQL, Text2SQLWithReasoning
