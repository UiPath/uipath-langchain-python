from unittest.mock import AsyncMock, MagicMock

from uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph import (
    QueryExecutor,
)


async def test_query_executor_forwards_source() -> None:
    """QueryExecutor forwards the caller-provided source as the query/execute
    header value — the low-code runtime passes LOW_CODE_AGENT, coded agents pass
    their own — so it is not hardcoded in this shared library."""
    entities = MagicMock()
    entities.query_entity_records_async = AsyncMock(return_value=[{"id": 1}])

    result = await QueryExecutor(entities, "LOW_CODE_AGENT")(
        "SELECT id FROM TaskEntity LIMIT 10"
    )

    entities.query_entity_records_async.assert_awaited_once_with(
        sql_query="SELECT id FROM TaskEntity LIMIT 10",
        source="LOW_CODE_AGENT",
    )
    assert result["records"] == [{"id": 1}]


async def test_query_executor_source_defaults_to_none() -> None:
    """With no source provided, none is sent (the SDK omits the header)."""
    entities = MagicMock()
    entities.query_entity_records_async = AsyncMock(return_value=[])

    await QueryExecutor(entities)("SELECT id FROM TaskEntity LIMIT 10")

    entities.query_entity_records_async.assert_awaited_once_with(
        sql_query="SELECT id FROM TaskEntity LIMIT 10",
        source=None,
    )
