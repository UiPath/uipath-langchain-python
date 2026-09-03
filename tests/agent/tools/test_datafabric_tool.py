from types import SimpleNamespace
from typing import Any, cast, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.core.feature_flags import FeatureFlags
from uipath.platform.entities import DataFabricEntityItem

import uipath_langchain.agent.tools as agent_tools
from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)
from uipath_langchain.agent.tools.context_tool import create_context_tool
from uipath_langchain.agent.tools.datafabric_tool.datafabric_tool import (
    ENTITY_V3_API_FF,
    DataFabricTextQueryHandler,
    create_datafabric_tool,
)


class _FakeCompiledGraph:
    def __init__(self, result_state):
        self._result_state = result_state

    async def ainvoke(self, _state):
        return self._result_state


def _entity() -> DataFabricEntityItem:
    return DataFabricEntityItem(
        id="1312e893-8295-f111-9b33-0022482a9eea",
        entity_key="agentTest",
        name="agentTest",
        folder_key="379fec63-62b1-41ec-b2fc-718f8f7dda3c",
        description="Agent test records",
    )


def test_coded_datafabric_factory_is_public():
    assert agent_tools.create_datafabric_tool is create_datafabric_tool
    assert not hasattr(agent_tools, "create_datafabric_query_tool")


def test_create_datafabric_tool_builds_directly_configured_tool():
    entity = _entity()
    tool = create_datafabric_tool(
        llm=MagicMock(),
        name="query_agent_test",
        description="Query the agentTest entity.",
        entities=[entity],
        base_system_prompt="Answer only from Data Fabric.",
    )

    assert tool.name == "query_agent_test"
    assert tool.description == "Query the agentTest entity."
    assert tool.metadata == {"tool_type": "datafabric_sql"}
    assert isinstance(tool, BaseUiPathStructuredTool)
    assert tool.coroutine is not None
    handler = cast(Any, tool.coroutine).__self__
    assert isinstance(handler, DataFabricTextQueryHandler)
    assert get_type_hints(tool.coroutine) == {"user_query": str, "return": str}
    assert handler._resource_description == "Query the agentTest entity."
    assert handler._base_system_prompt == "Answer only from Data Fabric."
    assert handler._entity_set == [_entity()]

    entity.name = "mutated-after-tool-creation"
    assert handler._entity_set[0].name == "agentTest"


def test_create_datafabric_tool_preserves_empty_entity_set_behavior():
    tool = create_datafabric_tool(
        llm=MagicMock(),
        name="query_datafabric",
        description="Query Data Fabric.",
        entities=[],
        base_system_prompt="Answer only from Data Fabric.",
    )

    assert tool.name == "query_datafabric"


def test_low_code_context_uses_resource_based_datafabric_query_factory():
    resource = AgentContextResourceConfig(
        name="Agent Test Data",
        description="Low-code Data Fabric context.",
        contextType="datafabricentityset",
        entitySet=[_entity()],
    )

    with patch(
        "uipath_langchain.agent.tools.context_tool._extract_system_prompt",
        return_value="Low-code system prompt.",
    ):
        tool = create_context_tool(resource, MagicMock())

    assert isinstance(tool, BaseUiPathStructuredTool)
    assert tool.name == "Agent_Test_Data"
    assert tool.description == (
        "Query the following Data Fabric entities using natural language:\n"
        "- agentTest: Agent test records\n"
        "Describe what data you need and the tool will translate it to SQL, "
        "execute the query, and return a natural language answer."
    )
    assert tool.coroutine is not None
    handler = cast(Any, tool.coroutine).__self__
    assert isinstance(handler, DataFabricTextQueryHandler)
    assert handler._resource_description == "Low-code Data Fabric context."
    assert handler._base_system_prompt == "Low-code system prompt."
    assert handler._entity_set == [_entity()]


@pytest.mark.asyncio
async def test_datafabric_handler_resolves_entities_lazily_once():
    entity = _entity()
    sdk = MagicMock()
    sdk.entities.resolve_entity_set_async = AsyncMock(
        return_value=SimpleNamespace(
            entities=[MagicMock()],
            entities_service=MagicMock(),
        )
    )
    compiled = _FakeCompiledGraph({"messages": []})
    handler = DataFabricTextQueryHandler(entity_set=[entity], llm=MagicMock())

    with (
        patch("uipath.platform.UiPath", return_value=sdk),
        patch(
            "uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph.DataFabricGraph.create",
            return_value=compiled,
        ) as create_graph,
    ):
        first = await handler._ensure_datafabric_graph()
        second = await handler._ensure_datafabric_graph()

    assert first is compiled
    assert second is compiled
    sdk.entities.resolve_entity_set_async.assert_awaited_once_with([entity])
    create_graph.assert_called_once()


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


def _sdk_with_both_resolvers() -> MagicMock:
    """A UiPath SDK mock exposing both the current and V3 resolution methods."""
    resolution = SimpleNamespace(entities=[MagicMock()], entities_service=MagicMock())
    sdk = MagicMock()
    sdk.entities.resolve_entity_set_async = AsyncMock(return_value=resolution)
    sdk.entities.resolve_entity_set_v3_async = AsyncMock(return_value=resolution)
    return sdk


@pytest.mark.asyncio
async def test_entity_resolution_uses_v1_when_v3_flag_disabled():
    """Default (flag off): resolve via the current method, never the V3 one."""
    entity = _entity()
    sdk = _sdk_with_both_resolvers()
    handler = DataFabricTextQueryHandler(entity_set=[entity], llm=MagicMock())

    # Set the flag off explicitly (programmatic value beats any
    # UIPATH_FEATURE_EnableEntityV3API env var) so the assertion is deterministic.
    FeatureFlags.configure_flags({ENTITY_V3_API_FF: False})
    try:
        with (
            patch("uipath.platform.UiPath", return_value=sdk),
            patch(
                "uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph.DataFabricGraph.create",
                return_value=_FakeCompiledGraph({"messages": []}),
            ),
        ):
            await handler._ensure_datafabric_graph()
    finally:
        FeatureFlags.reset_flags()

    sdk.entities.resolve_entity_set_async.assert_awaited_once_with([entity])
    sdk.entities.resolve_entity_set_v3_async.assert_not_called()


@pytest.mark.asyncio
async def test_entity_resolution_uses_v3_when_v3_flag_enabled():
    """Flag on: resolve via the dedicated V3 method, never the current one."""
    entity = _entity()
    sdk = _sdk_with_both_resolvers()
    handler = DataFabricTextQueryHandler(entity_set=[entity], llm=MagicMock())

    FeatureFlags.configure_flags({ENTITY_V3_API_FF: True})
    try:
        with (
            patch("uipath.platform.UiPath", return_value=sdk),
            patch(
                "uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph.DataFabricGraph.create",
                return_value=_FakeCompiledGraph({"messages": []}),
            ),
        ):
            await handler._ensure_datafabric_graph()
    finally:
        FeatureFlags.reset_flags()

    sdk.entities.resolve_entity_set_v3_async.assert_awaited_once_with([entity])
    sdk.entities.resolve_entity_set_async.assert_not_called()
