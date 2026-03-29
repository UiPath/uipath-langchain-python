"""SQL tool with syntax validation via sqlparse."""

from typing import Any

import sqlparse
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig

from .base_uipath_structured_tool import BaseUiPathStructuredTool


def validate_sql(sql: str) -> str | None:
    """Validate SQL syntax using sqlparse.

    Returns:
        Error string if invalid, None if valid.
    """
    parsed = sqlparse.parse(sql)
    if not parsed or not parsed[0].tokens:
        return "Empty or unparseable SQL query"
    return None


class SqlTool(BaseUiPathStructuredTool):
    """Structured tool with SQL syntax validation.

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
        error = validate_sql(sql_query)
        if error:
            raise ValueError(error)
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )
