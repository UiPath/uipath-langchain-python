from unittest.mock import AsyncMock

import pytest

from uipath_langchain.agent.tools.datafabric_query_tool import (
    DataFabricQueryTool,
    _normalize_sql,
    _validate_sql,
)
from uipath_langchain.agent.tools.datafabric_tool.models import (
    DataFabricExecuteSqlInput,
)


def test_normalize_sql_strips_single_trailing_semicolon():
    assert _normalize_sql("SELECT 1;") == "SELECT 1"
    assert _normalize_sql("  SELECT 1;  ") == "SELECT 1"


def test_validate_sql_rejects_multiple_statements():
    assert (
        _validate_sql("SELECT 1; SELECT 2") == "Multiple SQL statements are not allowed"
    )


@pytest.mark.asyncio
async def test_datafabric_query_tool_executes_normalized_sql():
    coroutine = AsyncMock(return_value={"ok": True})
    tool = DataFabricQueryTool(
        name="execute_sql",
        description="Execute SQL",
        args_schema=DataFabricExecuteSqlInput,
        coroutine=coroutine,
    )

    result = await tool.ainvoke({"sql_query": "SELECT 1;"})

    assert result == {"ok": True}
    coroutine.assert_awaited_once_with(sql_query="SELECT 1")


@pytest.mark.asyncio
async def test_datafabric_query_tool_rejects_multiple_statements():
    coroutine = AsyncMock(return_value={"ok": True})
    tool = DataFabricQueryTool(
        name="execute_sql",
        description="Execute SQL",
        args_schema=DataFabricExecuteSqlInput,
        coroutine=coroutine,
    )

    with pytest.raises(ValueError, match="Multiple SQL statements are not allowed"):
        await tool.ainvoke({"sql_query": "SELECT 1; SELECT 2"})

    coroutine.assert_not_awaited()
