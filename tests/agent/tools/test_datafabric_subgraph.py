from unittest.mock import AsyncMock, MagicMock

from uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph import (
    LOW_CODE_AGENT,
    QueryExecutor,
)


async def test_query_executor_sends_low_code_agent_source() -> None:
    """The agent tags its query/execute call as LOW_CODE_AGENT so FQS types
    relationship fields as their scalar id (enabling `rel = Other.Id` joins)."""
    entities = MagicMock()
    entities.query_entity_records_async = AsyncMock(return_value=[{"id": 1}])

    result = await QueryExecutor(entities)("SELECT id FROM TaskEntity LIMIT 10")

    entities.query_entity_records_async.assert_awaited_once_with(
        sql_query="SELECT id FROM TaskEntity LIMIT 10",
        source=LOW_CODE_AGENT,
    )
    assert result["records"] == [{"id": 1}]
