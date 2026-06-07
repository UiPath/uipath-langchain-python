from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from uipath_langchain.agent.tools.datafabric_tool.datafabric_tool import (
    DataFabricTextQueryHandler,
)


class _FakeCompiledGraph:
    def __init__(self, result_state):
        self._result_state = result_state

    async def ainvoke(self, _state):
        return self._result_state


@pytest.mark.asyncio
async def test_datafabric_handler_returns_single_terminal_tool_message():
    handler = DataFabricTextQueryHandler(
        entity_set=[],
        llm=MagicMock(),
    )
    handler._compiled = _FakeCompiledGraph(  # type: ignore[assignment]
        {
            "messages": [
                ToolMessage(
                    content="{'records': [1], 'total_count': 1}", tool_call_id="1"
                )
            ]
        }
    )

    result = await handler("count rows")

    assert result == "{'records': [1], 'total_count': 1}"


@pytest.mark.asyncio
async def test_datafabric_handler_aggregates_multiple_terminal_tool_messages():
    handler = DataFabricTextQueryHandler(
        entity_set=[],
        llm=MagicMock(),
    )
    handler._compiled = _FakeCompiledGraph(  # type: ignore[assignment]
        {
            "messages": [
                ToolMessage(
                    content="{'records': [{'id': 1}], 'total_count': 1, 'sql_query': 'SELECT ...'}",
                    tool_call_id="1",
                ),
                ToolMessage(
                    content="{'records': [{'name': 'Acme'}], 'total_count': 1, 'sql_query': 'SELECT ...'}",
                    tool_call_id="2",
                ),
            ]
        }
    )

    result = await handler("show id and name")

    assert "Multiple SQL queries executed successfully." in result
    assert "Result 1:" in result
    assert "Result 2:" in result
    assert (
        "{'records': [{'id': 1}], 'total_count': 1, 'sql_query': 'SELECT ...'}"
        in result
    )
    assert (
        "{'records': [{'name': 'Acme'}], 'total_count': 1, 'sql_query': 'SELECT ...'}"
        in result
    )


@pytest.mark.asyncio
async def test_datafabric_handler_prefers_terminal_ai_message():
    handler = DataFabricTextQueryHandler(
        entity_set=[],
        llm=MagicMock(),
    )
    handler._compiled = _FakeCompiledGraph(  # type: ignore[assignment]
        {
            "messages": [
                ToolMessage(
                    content="{'records': [], 'total_count': 0}", tool_call_id="1"
                ),
                AIMessage(content="I could not find any matching rows."),
            ]
        }
    )

    result = await handler("find missing row")

    assert result == "I could not find any matching rows."
