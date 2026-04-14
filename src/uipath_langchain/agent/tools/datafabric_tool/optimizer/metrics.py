"""Metrics for evaluating Text2SQL generation accuracy."""

import re
from typing import Any


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, collapse whitespace, strip."""
    sql = sql.strip().rstrip(";").strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    # Normalize spacing around commas and parentheses
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*\(\s*", "(", sql)
    sql = re.sub(r"\s*\)\s*", ")", sql)
    return sql


def _extract_clauses(sql: str) -> dict[str, str]:
    """Extract major SQL clauses for structural comparison."""
    normalized = _normalize_sql(sql)
    clauses: dict[str, str] = {}
    # Order matters — later keywords split the remaining string
    keywords = [
        "select",
        "from",
        "left join",
        "join",
        "where",
        "group by",
        "having",
        "order by",
        "limit",
        "offset",
    ]
    remaining = normalized
    for kw in keywords:
        parts = remaining.split(kw, 1)
        if len(parts) == 2:
            clauses[kw] = parts[1].strip()
            remaining = parts[1]
    return clauses


def sql_match_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score SQL generation accuracy.

    Returns:
        1.0 — exact match after normalization
        0.5 — same clauses present (structural match)
        0.0 — no match
    """
    expected = getattr(example, "sql", "")
    predicted = getattr(prediction, "sql", "")

    if not expected or not predicted:
        return 0.0

    # Exact match after normalization
    if _normalize_sql(expected) == _normalize_sql(predicted):
        return 1.0

    # Structural: same clause set present
    expected_clauses = _extract_clauses(expected)
    predicted_clauses = _extract_clauses(predicted)
    if expected_clauses.keys() == predicted_clauses.keys():
        return 0.5

    return 0.0
