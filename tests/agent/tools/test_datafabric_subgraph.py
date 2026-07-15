from unittest.mock import AsyncMock, MagicMock

from uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph import (
    QueryExecutor,
)


async def test_query_executor_requests_relationships_as_scalar() -> None:
    """The Data Fabric tool always requests scalar relationship typing so the SQL
    it writes can join on ``relationshipField = Other.Id``."""
    entities = MagicMock()
    entities.query_entity_records_async = AsyncMock(return_value=[{"id": 1}])

    result = await QueryExecutor(entities)("SELECT id FROM TaskEntity LIMIT 10")

    entities.query_entity_records_async.assert_awaited_once_with(
        sql_query="SELECT id FROM TaskEntity LIMIT 10",
        relationships_as_scalar=True,
    )
    assert result["records"] == [{"id": 1}]
