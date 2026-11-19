"""Tests for the ontology inner sub-graph (``OntologyGraph``).

The sub-graph is prompt-agnostic: the caller passes a pre-built ``system_prompt``
and the ``execute_sql`` tool bound to the Ontology service. Covers the tool
node's dispatch/terminal logic, routing and compile.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.constants import END

from uipath_langchain.agent.tools.datafabric_tool.models import (
    DataFabricExecuteSqlInput,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_subgraph import (
    OntologyGraph,
    OntologySubgraphState,
)


@pytest.fixture
def executor():
    return AsyncMock(
        return_value={"records": [{"x": 1}], "total_count": 1, "sql_query": "SELECT 1"}
    )


@pytest.fixture
def execute_sql_tool(executor):
    return StructuredTool.from_function(
        coroutine=executor,
        name="execute_sql",
        description="run",
        args_schema=DataFabricExecuteSqlInput,
    )


@pytest.fixture
def make_graph(execute_sql_tool):
    def _make(system_prompt="SYS"):
        return OntologyGraph(
            llm=MagicMock(),
            system_prompt=system_prompt,
            execute_sql_tool=execute_sql_tool,
        )

    return _make


def _tc(name, args=None, cid="c1"):
    return {"name": name, "args": args or {}, "id": cid, "type": "tool_call"}


def test_system_prompt_used_verbatim(make_graph):
    assert make_graph("MY_PROMPT")._system_message.content == "MY_PROMPT"


def test_binds_the_injected_tool(execute_sql_tool):
    llm = MagicMock()
    OntologyGraph(llm=llm, system_prompt="p", execute_sql_tool=execute_sql_tool)
    llm.model_copy.return_value.bind_tools.assert_called_once_with([execute_sql_tool])


async def test_execute_tool_call_sql_with_rows_is_terminal(make_graph):
    graph = make_graph()
    msg, ok = await graph._execute_tool_call(
        _tc("execute_sql", {"sql_query": "SELECT 1"})
    )
    assert ok is True
    assert msg.name == "execute_sql"
    assert "'x': 1" in msg.content


async def test_execute_tool_call_sql_no_rows_not_terminal(make_graph, executor):
    executor.return_value = {"records": [], "total_count": 0, "sql_query": "SELECT 1"}
    graph = make_graph()
    _, ok = await graph._execute_tool_call(
        _tc("execute_sql", {"sql_query": "SELECT 1"})
    )
    assert ok is False


async def test_execute_tool_call_error_dict_is_not_terminal(make_graph, executor):
    executor.return_value = {
        "records": [],
        "total_count": 0,
        "error": "400: unknown object",
        "sql_query": "SELECT 1",
    }
    graph = make_graph()
    msg, ok = await graph._execute_tool_call(
        _tc("execute_sql", {"sql_query": "SELECT 1"})
    )
    assert ok is False
    assert "unknown object" in msg.content


async def test_execute_tool_call_value_error_becomes_error_dict(make_graph):
    graph = make_graph()
    graph._execute_sql_tool = MagicMock()
    graph._execute_sql_tool.ainvoke = AsyncMock(side_effect=ValueError("bad sql"))
    msg, ok = await graph._execute_tool_call(_tc("execute_sql", {"sql_query": "x"}))
    assert ok is False
    assert "bad sql" in msg.content


async def test_tool_node_terminal_on_sql_rows(make_graph):
    graph = make_graph()
    ai = AIMessage(
        content="", tool_calls=[_tc("execute_sql", {"sql_query": "SELECT 1"}, "a")]
    )
    out = await graph.tool_node(OntologySubgraphState(messages=[ai], iteration_count=0))
    assert out["last_tool_success"] is True
    assert out["iteration_count"] == 1


def test_create_returns_compiled_graph(execute_sql_tool):
    compiled = OntologyGraph.create(
        llm=MagicMock(), system_prompt="SYS", execute_sql_tool=execute_sql_tool
    )
    assert hasattr(compiled, "ainvoke")


# --- graph nodes ------------------------------------------------------------


async def test_llm_node_invokes_inner_llm(make_graph):
    graph = make_graph()
    graph._inner_llm.ainvoke = AsyncMock(return_value=AIMessage(content="hi"))
    out = await graph.llm_node(
        OntologySubgraphState(messages=[HumanMessage(content="q")])
    )
    assert out["messages"][0].content == "hi"


async def test_tool_node_without_tool_calls_is_noop(make_graph):
    graph = make_graph()
    state = OntologySubgraphState(
        messages=[AIMessage(content="final")], iteration_count=3
    )
    assert await graph.tool_node(state) == {"iteration_count": 3}


async def test_termination_node_reports_attempts(make_graph):
    graph = make_graph()
    out = await graph.termination_node(OntologySubgraphState(iteration_count=7))
    assert "7 SQL attempts" in out["messages"][0].content


def test_router_to_tool_when_calls_under_limit(make_graph):
    graph = make_graph()
    ai = AIMessage(content="", tool_calls=[_tc("execute_sql", cid="a")])
    assert graph.router(OntologySubgraphState(messages=[ai], iteration_count=0)) == (
        "inner_tool"
    )


def test_router_to_termination_at_limit(make_graph):
    graph = make_graph()  # default max_iterations=25
    ai = AIMessage(content="", tool_calls=[_tc("execute_sql", cid="a")])
    assert graph.router(OntologySubgraphState(messages=[ai], iteration_count=25)) == (
        "termination"
    )


def test_router_ends_without_tool_calls(make_graph):
    graph = make_graph()
    assert graph.router(OntologySubgraphState(messages=[AIMessage(content="f")])) == END


def test_router_ends_without_messages(make_graph):
    assert make_graph().router(OntologySubgraphState(messages=[])) == END


def test_tool_router_ends_on_success(make_graph):
    assert (
        make_graph().tool_router(OntologySubgraphState(last_tool_success=True)) == END
    )


def test_tool_router_loops_on_failure(make_graph):
    state = OntologySubgraphState(last_tool_success=False)
    assert make_graph().tool_router(state) == "inner_llm"
