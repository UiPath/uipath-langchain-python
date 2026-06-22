"""Tests for the ontology additions to the Data Fabric inner sub-graph.

Covers: conditional binding of fetch_ontology, dispatch-by-name in
_execute_tool_call, and the any(...) terminal logic in tool_node.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from uipath_langchain.agent.tools.datafabric_tool import datafabric_subgraph as dsg
from uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph import (
    DataFabricGraph,
    DataFabricSubgraphState,
)


@pytest.fixture
def entities_service():
    es = MagicMock()
    es.query_entity_records_async = AsyncMock(return_value=[{"x": 1}])
    es.get_ontology_file_async = AsyncMock(
        return_value={"content": "OWLX", "mediaType": "text/turtle"}
    )
    return es


@pytest.fixture
def make_graph(monkeypatch, entities_service):
    # Isolate from the prompt builder; we only exercise tools/routing here.
    monkeypatch.setattr(dsg.datafabric_prompt_builder, "build", lambda *a, **k: "SYS")

    def _make(ontologies=None):
        return DataFabricGraph(
            llm=MagicMock(),
            entities=[],
            entities_service=entities_service,
            ontologies=ontologies,
        )

    return _make


def _tc(name, args=None, cid="c1"):
    return {"name": name, "args": args or {}, "id": cid, "type": "tool_call"}


def test_fetch_ontology_bound_only_when_ontologies(make_graph):
    without = make_graph(None)
    assert "execute_sql" in without._tools_by_name
    assert "fetch_ontology" not in without._tools_by_name

    with_onto = make_graph([("library", None)])
    assert "fetch_ontology" in with_onto._tools_by_name


async def test_execute_tool_call_unknown_tool(make_graph):
    graph = make_graph()
    msg, ok = await graph._execute_tool_call(_tc("does_not_exist"))
    assert ok is False
    assert "Unknown tool" in str(msg.content)


async def test_execute_tool_call_sql_with_rows_is_terminal(make_graph):
    graph = make_graph()
    msg, ok = await graph._execute_tool_call(
        _tc("execute_sql", {"sql_query": "SELECT 1"})
    )
    assert ok is True


async def test_execute_tool_call_sql_no_rows_not_terminal(make_graph, entities_service):
    entities_service.query_entity_records_async = AsyncMock(return_value=[])
    graph = make_graph()
    msg, ok = await graph._execute_tool_call(
        _tc("execute_sql", {"sql_query": "SELECT 1"})
    )
    assert ok is False


async def test_execute_tool_call_fetch_ontology_not_terminal(make_graph):
    graph = make_graph([("library", None)])
    msg, ok = await graph._execute_tool_call(_tc("fetch_ontology"))
    assert ok is False  # ontology fetch loops back, never terminal
    assert "library" in str(msg.content)


async def test_tool_node_any_succeeds_with_mixed_batch(make_graph):
    graph = make_graph([("library", None)])
    ai = AIMessage(
        content="",
        tool_calls=[
            _tc("execute_sql", {"sql_query": "SELECT 1"}, "a"),
            _tc("fetch_ontology", {}, "b"),
        ],
    )
    out = await graph.tool_node(DataFabricSubgraphState(messages=[ai]))
    # SQL returned rows → terminal, even though fetch_ontology (non-terminal)
    # was co-issued in the same turn. This is the all()->any() fix.
    assert out["last_tool_success"] is True
    assert len(out["messages"]) == 2


async def test_tool_node_not_terminal_when_only_ontology(make_graph):
    graph = make_graph([("library", None)])
    ai = AIMessage(content="", tool_calls=[_tc("fetch_ontology", {}, "b")])
    out = await graph.tool_node(DataFabricSubgraphState(messages=[ai]))
    assert out["last_tool_success"] is False


async def test_execute_tool_call_sql_value_error_becomes_error_dict(make_graph):
    # execute_sql raises ValueError on multiple statements; it must be caught and
    # turned into an error result (non-terminal), not propagated.
    graph = make_graph()
    msg, ok = await graph._execute_tool_call(
        _tc("execute_sql", {"sql_query": "SELECT 1; SELECT 2"})
    )
    assert ok is False
    assert "error" in str(msg.content)


def test_create_returns_compiled_graph(monkeypatch, entities_service):
    monkeypatch.setattr(dsg.datafabric_prompt_builder, "build", lambda *a, **k: "SYS")
    compiled = DataFabricGraph.create(
        llm=MagicMock(),
        entities=[],
        entities_service=entities_service,
        ontologies=[("library", None)],
    )
    assert hasattr(compiled, "ainvoke")
