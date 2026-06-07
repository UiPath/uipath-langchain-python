"""Data Fabric query tool with SQL syntax validation via sqlparse."""

from typing import Any

import sqlparse
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig

from .base_uipath_structured_tool import BaseUiPathStructuredTool


def _normalize_sql(sql: str) -> str:
    """Normalize a generated SQL query before validation / execution.

    Strips surrounding whitespace and removes a single trailing semicolon so the
    backend receives one bare statement even if the model emits canonical SQL
    terminators by default.
    """
    normalized = sql.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def _validate_sql(sql: str) -> str | None:
    """Validate SQL syntax using sqlparse.

    Returns:
        Error string if invalid, None if valid.
    """
    parsed = sqlparse.parse(sql)
    if not parsed or not parsed[0].tokens:
        return "Empty or unparseable SQL query"
    if len(parsed) != 1:
        return "Multiple SQL statements are not allowed"
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
        normalized_sql_query = _normalize_sql(sql_query)
        error = _validate_sql(normalized_sql_query)
        if error:
            raise ValueError(error)
        if "sql_query" in kwargs:
            kwargs["sql_query"] = normalized_sql_query
        elif args:
            args = (normalized_sql_query, *args[1:])
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )
