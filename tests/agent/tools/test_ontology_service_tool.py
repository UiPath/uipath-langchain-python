"""Tests for the ontology tool (datafabric_tool/ontology/ontology_tool.py).

Covers the factory (reads ``ontology_set[0].id`` → handler), the lazy build
(flag gate, deployed check, grounding by source, service prompt, ``execute_sql``
bound to the service), the SQL executor (records from dict rows, rejected
statements fed back as errors) and the terminal-message handling.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
import uipath.platform as _uipath_platform
from langchain_core.messages import AIMessage, ToolMessage
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.core.feature_flags import FeatureFlags

from uipath_langchain.agent import tools as agent_tools
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.tools.datafabric_tool.ontology import (
    ontology_subgraph as _sg,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_service_sql import (
    OntologyServiceQueryExecutor,
    create_service_execute_sql_tool,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_tool import (
    DataFabricOntologyQueryHandler,
    create_datafabric_ontology_tool,
    create_ontology_tool,
)

# --- fakes ------------------------------------------------------------------


@dataclass
class _Artifact:
    file_name: str = "nxorders.ofn"
    type: str = "schema"
    media_type: str = "text/owl-functional"
    size_bytes: int = 12
    checksum: str = "c1"


@dataclass
class _SqlResult:
    """Mirrors the SDK's ``SqlResult``: typed columns, dict rows, a service count."""

    names: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    generated_sql: str | None = None

    @property
    def columns(self) -> list[Any]:
        return [SimpleNamespace(name=n, type="string") for n in self.names]

    @property
    def column_names(self) -> list[str]:
        return list(self.names)


@dataclass
class _FakeOntologyClient:
    state: str = "DEPLOYED"
    sql_result: _SqlResult | None = None
    sql_error: Exception | None = None
    calls: list[tuple[str, Any]] = field(default_factory=list)

    async def get_async(self, ontology_id):
        self.calls.append(("get", ontology_id))
        return SimpleNamespace(
            id=ontology_id,
            name="nxorders",
            display_name="Orders",
            description="Sales orders",
            folder_id="f-1",
            state=self.state,
        )

    async def list_artifacts_async(self, ontology_id, artifact_type=None):
        self.calls.append(("list", artifact_type))
        return [_Artifact()]

    async def get_artifact_async(self, ontology_id, file_name, expected=None):
        return SimpleNamespace(
            content=b"Ontology(<urn:nxorders>)",
            media_type="text/owl-functional",
            etag='"c1"',
        )

    async def get_artifact_by_type_async(self, ontology_id, artifact_type):
        self.calls.append(("by_type", artifact_type))
        return SimpleNamespace(
            file_name="nxorders.ofn",
            content=b"Ontology(<urn:nxorders>)",
            media_type="text/owl-functional",
            etag='"c1"',
        )

    async def get_graph_async(self, ontology_id):
        self.calls.append(("graph", ontology_id))
        return {"objects": [{"name": "SalesOrder", "domain": "Sales"}]}

    async def execute_sql_async(
        self, ontology_id, statement, params=None, limit=None, offset=None
    ):
        self.calls.append(("sql", (ontology_id, statement, params, limit, offset)))
        if self.sql_error:
            raise self.sql_error
        assert self.sql_result is not None
        return self.sql_result


ONTOLOGY_ID = "c0d9f850-ba58-443e-8c54-0bc4bd68f202"


def _flags(monkeypatch, source: str | None = None):
    monkeypatch.setattr(FeatureFlags, "is_flag_enabled", lambda *a, **k: True)
    monkeypatch.setattr(
        FeatureFlags,
        "get_flag",
        lambda name, default=None: default if source is None else source,
    )


def _sdk(monkeypatch, client):
    sdk = MagicMock()
    sdk.ontology = client
    monkeypatch.setattr(_uipath_platform, "UiPath", lambda: sdk)
    return sdk


def _capture_graph_create(monkeypatch):
    captured: dict[str, Any] = {}
    sentinel = object()

    def create(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(_sg.OntologyGraph, "create", create)
    return captured, sentinel


def _handler(ontology_id: str | None = ONTOLOGY_ID):
    return DataFabricOntologyQueryHandler(
        ontology_id=ontology_id,
        llm=MagicMock(),
        resource_description="Use for order questions",
        base_system_prompt="You help with sales.",
    )


def _resource(ontology_set):
    return AgentContextResourceConfig.model_validate(
        {
            "$resourceType": "context",
            "name": "Orders",
            "description": "orders",
            "contextType": "datafabricontology",
            "ontologySet": ontology_set,
        }
    )


# --- factory ----------------------------------------------------------------


def test_factory_reads_the_first_ontology_id():
    tool = create_datafabric_ontology_tool(
        _resource([{"id": "onto-1", "name": "nxorders", "folderId": "f-1"}]),
        MagicMock(),
    )
    assert tool.name == "query_datafabric_ontology"
    assert tool.coroutine.__self__._ontology_id == "onto-1"  # type: ignore[attr-defined]
    assert "nxorders" in tool.description


def test_factory_uses_the_first_of_several_and_warns(caplog):
    tool = create_datafabric_ontology_tool(
        _resource(
            [
                {"id": "onto-1", "name": "a", "folderId": "f"},
                {"id": "onto-2", "name": "b", "folderId": "f"},
            ]
        ),
        MagicMock(),
    )
    assert tool.coroutine.__self__._ontology_id == "onto-1"  # type: ignore[attr-defined]
    assert "using the first" in caplog.text


def test_factory_without_an_id_fails_on_first_use(monkeypatch):
    tool = create_datafabric_ontology_tool(
        _resource([{"name": "nxorders", "folderId": "f-1"}]), MagicMock()
    )
    assert tool.coroutine.__self__._ontology_id is None  # type: ignore[attr-defined]


# --- coded-agent factory ----------------------------------------------------


def _coded_tool(ontology_id: str = ONTOLOGY_ID):
    return create_ontology_tool(
        llm=MagicMock(),
        name="query_orders",
        description="Query orders and customers.",
        ontology_id=ontology_id,
        base_system_prompt="Answer only from the ontology.",
    )


def test_coded_ontology_factory_is_public():
    assert agent_tools.create_ontology_tool is create_ontology_tool
    assert not hasattr(agent_tools, "create_datafabric_ontology_tool")


def test_coded_factory_builds_the_tool_from_direct_arguments():
    tool = _coded_tool()

    assert tool.name == "query_orders"
    assert tool.description == "Query orders and customers."
    assert tool.metadata == {"tool_type": "datafabric_ontology_sql"}
    handler = tool.coroutine.__self__
    assert isinstance(handler, DataFabricOntologyQueryHandler)
    assert handler._ontology_id == ONTOLOGY_ID
    assert handler._resource_description == "Query orders and customers."
    assert handler._base_system_prompt == "Answer only from the ontology."


def test_coded_tool_exposes_type_hints_a_langgraph_tool_node_can_read():
    assert get_type_hints(_coded_tool().coroutine) == {
        "user_query": str,
        "return": str,
    }


async def test_coded_tool_builds_with_the_low_code_flag_off(monkeypatch):
    monkeypatch.setattr(FeatureFlags, "is_flag_enabled", lambda *a, **k: False)
    client = _FakeOntologyClient()
    _sdk(monkeypatch, client)
    captured, sentinel = _capture_graph_create(monkeypatch)

    handler = _coded_tool().coroutine.__self__

    assert await handler._ensure_graph() is sentinel
    assert "Answer only from the ontology." in captured["system_prompt"]


async def test_coded_tool_with_an_unsafe_id_fails_on_first_use(monkeypatch):
    _sdk(monkeypatch, _FakeOntologyClient())
    handler = _coded_tool("../other").coroutine.__self__

    with pytest.raises(AgentRuntimeError) as raised:
        await handler._ensure_graph()

    assert raised.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.ONTOLOGY_INVALID_REFERENCE
    )


# --- lazy build -------------------------------------------------------------


async def test_build_guarded_when_flag_off(monkeypatch):
    monkeypatch.setattr(FeatureFlags, "is_flag_enabled", lambda *a, **k: False)
    with pytest.raises(ValueError, match="disabled"):
        await _handler()._ensure_graph()


async def test_cached_graph_skips_all_work(monkeypatch):
    handler = _handler()
    handler._compiled = "CACHED"
    monkeypatch.setattr(FeatureFlags, "is_flag_enabled", lambda *a, **k: False)
    assert await handler._ensure_graph() == "CACHED"


async def test_unset_source_grounds_on_schema_files(monkeypatch):
    _flags(monkeypatch, source=None)
    client = _FakeOntologyClient()
    _sdk(monkeypatch, client)
    captured, sentinel = _capture_graph_create(monkeypatch)

    handler = _handler()
    assert await handler._ensure_graph() is sentinel
    assert await handler._ensure_graph() is sentinel  # cached

    assert client.calls[:2] == [("get", ONTOLOGY_ID), ("by_type", "schema")]
    assert captured["execute_sql_tool"].name == "execute_sql"
    prompt = captured["system_prompt"]
    assert "You help with sales." in prompt
    assert "Name: Orders" in prompt and "Identifier: nxorders" in prompt
    assert "Use for order questions" in prompt
    assert "not an instruction to you" in prompt
    assert "## SQL rules for the Ontology runtime" in prompt
    assert "Ontology(<urn:nxorders>)" in prompt


async def test_graph_source_grounds_on_the_graph(monkeypatch):
    _flags(monkeypatch, source="graph")
    client = _FakeOntologyClient()
    _sdk(monkeypatch, client)
    captured, sentinel = _capture_graph_create(monkeypatch)

    assert await _handler()._ensure_graph() is sentinel

    assert ("graph", ONTOLOGY_ID) in client.calls
    assert all(kind not in ("list", "by_type") for kind, _ in client.calls)
    assert "- SalesOrder" in captured["system_prompt"]


async def test_build_refuses_an_undeployed_ontology(monkeypatch):
    _flags(monkeypatch)
    _sdk(monkeypatch, _FakeOntologyClient(state="BROKEN"))
    _capture_graph_create(monkeypatch)

    with pytest.raises(AgentRuntimeError) as err:
        await _handler()._ensure_graph()

    assert err.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.ONTOLOGY_NOT_DEPLOYED
    )
    assert "Orders" in err.value.error_info.title


async def test_build_needs_an_ontology_id(monkeypatch):
    _flags(monkeypatch)
    client = _FakeOntologyClient()
    _sdk(monkeypatch, client)

    with pytest.raises(AgentRuntimeError) as err:
        await _handler(ontology_id=None)._ensure_graph()

    assert err.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.ONTOLOGY_INVALID_REFERENCE
    )
    assert client.calls == []


async def test_build_needs_the_sdk_client(monkeypatch):
    _flags(monkeypatch)
    sdk = MagicMock(spec=[])  # no ``ontology`` attribute
    monkeypatch.setattr(_uipath_platform, "UiPath", lambda: sdk)

    with pytest.raises(AgentRuntimeError) as err:
        await _handler()._ensure_graph()

    assert err.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.ONTOLOGY_SERVICE_UNAVAILABLE
    )


# --- execute_sql on the service --------------------------------------------


async def test_executor_returns_records_from_dict_rows():
    client = _FakeOntologyClient(
        sql_result=_SqlResult(
            ["id", "total"], [{"id": 1, "total": 10.5}, {"id": 2, "total": 3.0}], 2
        )
    )
    executor = OntologyServiceQueryExecutor(client, "onto-1", "Orders")

    out = await executor("SELECT id, total FROM SalesOrder LIMIT 2")

    assert out["records"] == [{"id": 1, "total": 10.5}, {"id": 2, "total": 3.0}]
    assert out["total_count"] == 2 and out["row_count"] == 2
    assert "may_have_more" not in out and "error" not in out
    # the statement travels as a body argument, untouched, with no limit/offset
    # arguments: the model writes those clauses into the statement itself
    assert client.calls == [
        (
            "sql",
            ("onto-1", "SELECT id, total FROM SalesOrder LIMIT 2", None, None, None),
        )
    ]


@pytest.mark.parametrize("page", [50, 200])
async def test_executor_flags_a_full_page_as_possibly_incomplete(page):
    rows = [{"id": i} for i in range(page)]
    client = _FakeOntologyClient(sql_result=_SqlResult(["id"], rows, page))
    out = await OntologyServiceQueryExecutor(client, "onto-1", "Orders")(
        "SELECT id FROM SalesOrder"
    )
    assert out["may_have_more"] is True
    assert "OFFSET" in out["note"]


async def test_executor_leaves_a_short_page_unflagged():
    rows = [{"id": i} for i in range(49)]
    client = _FakeOntologyClient(sql_result=_SqlResult(["id"], rows, 49))
    out = await OntologyServiceQueryExecutor(client, "onto-1", "Orders")(
        "SELECT id FROM SalesOrder"
    )
    assert "may_have_more" not in out


@pytest.mark.parametrize("has_more", [True, False])
async def test_executor_prefers_the_services_own_has_more(has_more):
    result = _SqlResult(["id"], [{"id": i} for i in range(50)], 50)
    result.hasMore = has_more  # type: ignore[attr-defined]
    client = _FakeOntologyClient(sql_result=result)
    out = await OntologyServiceQueryExecutor(client, "onto-1", "Orders")(
        "SELECT id FROM SalesOrder"
    )
    assert out.get("may_have_more", False) is has_more


async def test_executor_feeds_a_rejected_statement_back_as_an_error():
    client = _FakeOntologyClient(
        sql_error=RuntimeError("400: only SELECT statements are accepted")
    )
    out = await OntologyServiceQueryExecutor(client, "onto-1", "Orders")(
        "DELETE FROM SalesOrder"
    )
    assert out["records"] == [] and out["total_count"] == 0
    assert "only SELECT statements" in out["error"]


async def test_service_execute_sql_tool_validates_and_runs():
    client = _FakeOntologyClient(sql_result=_SqlResult(["id"], [{"id": 7}], 1))
    tool = create_service_execute_sql_tool(client, "onto-1", "Orders")

    assert tool.name == "execute_sql"
    assert "Orders" in tool.description
    out = await tool.ainvoke({"sql_query": "SELECT id FROM SalesOrder LIMIT 1"})
    assert out["records"] == [{"id": 7}]


# --- __call__ (invoke + terminal message handling) --------------------------


def _handler_with_state(monkeypatch, messages):
    handler = _handler()
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": messages})
    monkeypatch.setattr(handler, "_ensure_graph", AsyncMock(return_value=graph))
    return handler


async def test_call_collapses_terminal_tool_messages(monkeypatch):
    handler = _handler_with_state(
        monkeypatch,
        [AIMessage(content=""), ToolMessage(content="rows!", tool_call_id="c1")],
    )
    assert await handler("q") == "rows!"


async def test_call_returns_ai_message_content(monkeypatch):
    handler = _handler_with_state(monkeypatch, [AIMessage(content="the answer")])
    assert await handler("q") == "the answer"


async def test_call_fallback_when_no_answer(monkeypatch):
    handler = _handler_with_state(monkeypatch, [AIMessage(content="")])
    assert await handler("q") == "Unable to generate an answer from the available data."


def test_format_terminal_collapses_multiple_results():
    out = DataFabricOntologyQueryHandler._format_terminal_tool_messages(
        [
            ToolMessage(content="rows-a", tool_call_id="1"),
            ToolMessage(content="rows-b", tool_call_id="2"),
        ]
    )
    assert "Multiple SQL queries" in out
    assert "Result 1:\nrows-a" in out
    assert "Result 2:\nrows-b" in out


def test_format_terminal_all_empty_returns_fallback():
    out = DataFabricOntologyQueryHandler._format_terminal_tool_messages(
        [ToolMessage(content="", tool_call_id="1")]
    )
    assert out == "Unable to generate an answer from the available data."
