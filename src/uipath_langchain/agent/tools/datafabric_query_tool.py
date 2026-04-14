"""Data Fabric query tool with SQL syntax and structural validation."""

import re
from typing import Any

import sqlparse
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig

from .base_uipath_structured_tool import BaseUiPathStructuredTool

_SELECT_STAR_RE = re.compile(r"\bSELECT\s+\*\s+FROM\b", re.IGNORECASE)
_COUNT_STAR_RE = re.compile(r"\bCOUNT\s*\(\s*(\*|1)\s*\)", re.IGNORECASE)
_UNSUPPORTED_JOIN_RE = re.compile(
    r"\b(RIGHT\s+JOIN|FULL\s+OUTER\s+JOIN|CROSS\s+JOIN)\b", re.IGNORECASE
)
_DML_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE)\b", re.IGNORECASE
)
_HAS_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_HAS_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)


def _validate_sql(sql: str) -> str | None:
    """Validate SQL syntax and structure.

    Performs sqlparse parseability check followed by structural checks
    that mirror backend constraints, giving faster local feedback.

    Returns:
        Error string if invalid, None if valid.
    """
    parsed = sqlparse.parse(sql)
    if not parsed or not parsed[0].tokens:
        return "Empty or unparseable SQL query"

    # DML/DDL rejection
    if _DML_RE.search(sql):
        return "Only SELECT queries are supported. INSERT/UPDATE/DELETE/DDL are not allowed."

    # SELECT * rejection
    if _SELECT_STAR_RE.search(sql):
        return "SELECT * is not allowed. Use explicit column names instead of SELECT *."

    # COUNT(*) / COUNT(1) rejection
    if _COUNT_STAR_RE.search(sql):
        return "COUNT(*) and COUNT(1) are not allowed. Use COUNT(column_name) instead."

    # Unsupported JOIN types
    match = _UNSUPPORTED_JOIN_RE.search(sql)
    if match:
        return f"{match.group(0).upper()} is not supported. Only LEFT JOIN is allowed."

    # Missing LIMIT when no WHERE clause
    if not _HAS_WHERE_RE.search(sql) and not _HAS_LIMIT_RE.search(sql):
        return "Queries without a WHERE clause must include a LIMIT clause (e.g. LIMIT 100)."

    return None


class DataFabricQueryTool(BaseUiPathStructuredTool):
    """Data Fabric query tool with SQL syntax validation.

    Validates that the input SQL is parseable before delegating
    to the underlying coroutine. On validation failure, raises
    a ValueError so the caller can handle it as needed.
    """

    async def _arun(
        __obj_internal_self__,
        *args: Any,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        sql_query = kwargs.get("sql_query") or (args[0] if args else "")
        error = _validate_sql(sql_query)
        if error:
            raise ValueError(error)
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )
