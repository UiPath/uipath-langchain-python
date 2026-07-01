"""Tests for the ontology handling in the Data Fabric inner sub-graph.

The ontology is fetched deterministically upstream and embedded in the inner
system prompt; the sub-graph itself only ever binds ``execute_sql``. Covers:
only execute_sql is bound, the ontology text is threaded into the prompt, and
dispatch/terminal logic in the tool node.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from uipath_langchain.agent.tools.datafabric_tool import datafabric_prompt_builder
from uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph import (
    DataFabricGraph,
    DataFabricSubgraphState,
)


@pytest.fixture
def entities_service():
    es = MagicMock()
    es.query_entity_records_async = AsyncMock(return_value=[{"x": 1}])
    return es


@pytest.fixture
def make_graph(monkeypatch, entities_service):
    # Isolate from the prompt builder; we only exercise tools/routing here.
    monkeypatch.setattr(datafabric_prompt_builder, "build", lambda *a, **k: "SYS")

    def _make(ontology_text=""):
        return DataFabricGraph(
            llm=MagicMock(),
            entities=[],
            entities_service=entities_service,
            ontology_text=ontology_text,
        )

    return _make


def _tc(name, args=None, cid="c1"):
    return {"name": name, "args": args or {}, "id": cid, "type": "tool_call"}


def test_ontology_text_threaded_into_prompt(monkeypatch, entities_service):
    captured: dict = {}
    monkeypatch.setattr(
        datafabric_prompt_builder,
        "build",
        lambda *a, **k: captured.update(k) or "SYS",
    )
    DataFabricGraph(
        llm=MagicMock(),
        entities=[],
        entities_service=entities_service,
        ontology_text="ONTOLOGY_XYZ",
    )
    assert captured.get("ontology_text") == "ONTOLOGY_XYZ"


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


async def test_execute_tool_call_sql_value_error_becomes_error_dict(make_graph):
    # execute_sql raises ValueError on multiple statements; it must be caught and
    # turned into an error result (non-terminal), not propagated.
    graph = make_graph()
    msg, ok = await graph._execute_tool_call(
        _tc("execute_sql", {"sql_query": "SELECT 1; SELECT 2"})
    )
    assert ok is False
    assert "error" in str(msg.content)


async def test_tool_node_terminal_on_sql_rows(make_graph):
    graph = make_graph()
    ai = AIMessage(
        content="", tool_calls=[_tc("execute_sql", {"sql_query": "SELECT 1"}, "a")]
    )
    out = await graph.tool_node(DataFabricSubgraphState(messages=[ai]))
    assert out["last_tool_success"] is True
    assert len(out["messages"]) == 1
    assert out["messages"][0].name == "execute_sql"


def test_create_returns_compiled_graph(monkeypatch, entities_service):
    monkeypatch.setattr(datafabric_prompt_builder, "build", lambda *a, **k: "SYS")
    compiled = DataFabricGraph.create(
        llm=MagicMock(),
        entities=[],
        entities_service=entities_service,
        ontology_text="onto",
    )
    assert hasattr(compiled, "ainvoke")
