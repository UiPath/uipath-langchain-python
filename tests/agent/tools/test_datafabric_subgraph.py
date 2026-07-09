"""Tests for the Data Fabric sub-graph module."""

from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.constants import END
from uipath.platform.entities import Entity
from uipath.platform.errors import EnrichedException

from uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph import (
    CATEGORY_MARKER,
    DataFabricGraph,
    DataFabricSubgraphState,
    QueryExecutor,
    _noop_context,
)

# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------


class _FakeCategory(Enum):
    BAD_SQL = "bad_sql"
    RETRYABLE = "retryable"
    UNKNOWN = "unknown"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class _FakeDataFabricError:
    code: str | None
    message: str | None
    trace_id: str | None
    category: _FakeCategory
    is_retryable: bool = False
    is_bad_sql: bool = False


def _make_entity(name: str, external_fields: list[str] | None = None) -> Entity:
    """Create a minimal Entity for testing."""
    e = MagicMock(spec=Entity)
    e.name = name
    e.external_fields = external_fields
    return e


def _make_entities_service(
    records: list[dict[str, Any]] | None = None,
    error: Exception | None = None,
) -> MagicMock:
    svc = MagicMock()
    if error:
        svc.query_entity_records_async = AsyncMock(side_effect=error)
    else:
        svc.query_entity_records_async = AsyncMock(return_value=records or [])
    return svc


# ---------------------------------------------------------------------------
# _noop_context
# ---------------------------------------------------------------------------


def test_noop_context_yields_none() -> None:
    with _noop_context() as val:
        assert val is None


# ---------------------------------------------------------------------------
# DataFabricSubgraphState
# ---------------------------------------------------------------------------


def test_state_defaults() -> None:
    state = DataFabricSubgraphState()
    assert state.messages == []
    assert state.iteration_count == 0
    assert state.last_tool_success is False
    assert state.last_error_category == ""
    assert state.last_error_detail == ""


# ---------------------------------------------------------------------------
# QueryExecutor.__init__  — entity attribute computation
# ---------------------------------------------------------------------------


def test_query_executor_entity_attrs_native_only() -> None:
    entities = [_make_entity("Orders"), _make_entity("Products")]
    svc = _make_entities_service()
    qe = QueryExecutor(svc, entities)
    assert qe._entity_attrs["df.entity_count"] == 2
    assert qe._entity_attrs["df.native_entity_count"] == 2
    assert qe._entity_attrs["df.federated_entity_count"] == 0
    assert "Orders" in str(qe._entity_attrs["df.native_entities"])
    assert qe._entity_attrs["df.federated_entities"] == ""


def test_query_executor_entity_attrs_mixed() -> None:
    entities = [
        _make_entity("Orders"),
        _make_entity("ExtTable", external_fields=["col1"]),
    ]
    svc = _make_entities_service()
    qe = QueryExecutor(svc, entities)
    assert qe._entity_attrs["df.native_entity_count"] == 1
    assert qe._entity_attrs["df.federated_entity_count"] == 1
    assert "ExtTable" in str(qe._entity_attrs["df.federated_entities"])


# ---------------------------------------------------------------------------
# QueryExecutor.__call__  — success path (no OTEL)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_executor_success_no_otel() -> None:
    records = [{"id": 1}, {"id": 2}]
    svc = _make_entities_service(records=records)
    qe = QueryExecutor(svc, [_make_entity("T")])

    with patch.dict(
        "sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}
    ):
        result = await qe("SELECT * FROM T")

    assert result["records"] == records
    assert result["total_count"] == 2
    assert result["sql_query"] == "SELECT * FROM T"
    assert "error" not in result


# ---------------------------------------------------------------------------
# QueryExecutor.__call__  — success path with OTEL span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_executor_success_with_span() -> None:
    records = [{"id": 1}]
    svc = _make_entities_service(records=records)
    qe = QueryExecutor(svc, [_make_entity("T")])

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
        return_value=False
    )

    with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
        result = await qe("SELECT 1")

    assert result["total_count"] == 1
    set_attr_calls = {
        call.args[0]: call.args[1] for call in mock_span.set_attribute.call_args_list
    }
    assert set_attr_calls["df.record_count"] == 1
    assert set_attr_calls["df.success"] is True


# ---------------------------------------------------------------------------
# QueryExecutor.__call__  — error path (no OTEL, plain exception)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_executor_error_no_otel() -> None:
    svc = _make_entities_service(error=RuntimeError("connection timeout"))
    qe = QueryExecutor(svc, [_make_entity("T")])

    with patch.dict(
        "sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}
    ):
        result = await qe("SELECT * FROM T")

    assert result["records"] == []
    assert result["total_count"] == 0
    assert "connection timeout" in result["error"]


# ---------------------------------------------------------------------------
# QueryExecutor.__call__  — error path with EnrichedException + DataFabricError
# ---------------------------------------------------------------------------


def _make_enriched_exception(msg: str = "enriched error") -> EnrichedException:
    """Create an EnrichedException with mocked httpx internals."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.headers = {"content-type": "application/json"}
    mock_response.content = msg.encode("utf-8")
    mock_request = MagicMock()
    mock_request.url = "https://datafabric_.example.com/query"
    mock_request.method = "POST"
    mock_error = MagicMock()
    mock_error.response = mock_response
    mock_error.request = mock_request
    return EnrichedException(mock_error)


@pytest.mark.asyncio
async def test_query_executor_error_with_datafabric_error() -> None:
    enriched = _make_enriched_exception()

    svc = _make_entities_service(error=enriched)
    qe = QueryExecutor(svc, [_make_entity("T")])

    fake_df_error = _FakeDataFabricError(
        code="SQL_VALIDATION",
        message="Invalid column reference",
        trace_id="abc123",
        category=_FakeCategory.BAD_SQL,
        is_bad_sql=True,
    )

    with (
        patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}),
        patch(
            "uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph.DataFabricError"
        ) as mock_dfe_cls,
    ):
        mock_dfe_cls.from_enriched_exception.return_value = fake_df_error
        result = await qe("SELECT bad_col FROM T")

    assert result["records"] == []
    assert "[SQL_VALIDATION]" in result["error"]
    assert "Fix the SQL syntax" in result["error"]


# ---------------------------------------------------------------------------
# QueryExecutor._build_error_detail
# ---------------------------------------------------------------------------


def test_build_error_detail_no_df_error() -> None:
    detail = QueryExecutor._build_error_detail(RuntimeError("boom"), None)
    assert detail == "boom"


def test_build_error_detail_with_code_and_bad_sql() -> None:
    df_err = _FakeDataFabricError(
        code="SQL_VALIDATION",
        message="Invalid column",
        trace_id=None,
        category=_FakeCategory.BAD_SQL,
        is_bad_sql=True,
    )
    detail = QueryExecutor._build_error_detail(RuntimeError("x"), df_err)  # type: ignore[arg-type]
    assert "[SQL_VALIDATION]" in detail
    assert "(category: bad_sql)" in detail
    assert "Invalid column" in detail
    assert "Fix the SQL syntax" in detail


def test_build_error_detail_retryable() -> None:
    df_err = _FakeDataFabricError(
        code="TIMEOUT",
        message="Request timed out",
        trace_id="t1",
        category=_FakeCategory.RETRYABLE,
        is_retryable=True,
    )
    detail = QueryExecutor._build_error_detail(RuntimeError("x"), df_err)  # type: ignore[arg-type]
    assert "[TIMEOUT]" in detail
    assert "transient" in detail


def test_build_error_detail_unknown_category() -> None:
    df_err = _FakeDataFabricError(
        code="SOMETHING",
        message="msg",
        trace_id=None,
        category=_FakeCategory.UNKNOWN,
    )
    detail = QueryExecutor._build_error_detail(RuntimeError("x"), df_err)  # type: ignore[arg-type]
    assert "[SOMETHING]" in detail
    # "unknown" category should NOT appear
    assert "(category:" not in detail


def test_build_error_detail_no_code() -> None:
    df_err = _FakeDataFabricError(
        code=None,
        message="msg",
        trace_id=None,
        category=_FakeCategory.UNKNOWN,
    )
    # Falls through to str(exc)
    detail = QueryExecutor._build_error_detail(RuntimeError("fallback"), df_err)  # type: ignore[arg-type]
    assert detail == "fallback"


def test_build_error_detail_no_message() -> None:
    df_err = _FakeDataFabricError(
        code="ERR",
        message=None,
        trace_id=None,
        category=_FakeCategory.INFRASTRUCTURE,
    )
    detail = QueryExecutor._build_error_detail(RuntimeError("x"), df_err)  # type: ignore[arg-type]
    assert "[ERR]" in detail
    assert "(category: infrastructure)" in detail


# ---------------------------------------------------------------------------
# DataFabricGraph — routing
# ---------------------------------------------------------------------------


def _make_graph() -> DataFabricGraph:
    """Create a DataFabricGraph with a mocked LLM."""
    llm = MagicMock(spec=["model_copy"])
    bound = MagicMock()
    copy = MagicMock()
    copy.bind_tools = MagicMock(return_value=bound)
    llm.model_copy.return_value = copy

    entities = [_make_entity("Orders")]
    svc = _make_entities_service()

    with patch(
        "uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph.datafabric_prompt_builder"
    ) as mock_pb:
        mock_pb.build.return_value = "system prompt"
        graph = DataFabricGraph(llm, entities, svc, max_iterations=3)
    return graph


def test_router_to_tool() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(
        messages=[
            AIMessage(
                content="", tool_calls=[{"id": "1", "name": "execute_sql", "args": {}}]
            )
        ],
        iteration_count=0,
    )
    assert graph.router(state) == "inner_tool"


def test_router_to_termination() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(
        messages=[
            AIMessage(
                content="", tool_calls=[{"id": "1", "name": "execute_sql", "args": {}}]
            )
        ],
        iteration_count=3,  # at max
    )
    assert graph.router(state) == "termination"


def test_router_to_end_no_tool_calls() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(
        messages=[AIMessage(content="final answer")],
    )
    assert graph.router(state) == END


def test_router_to_end_empty_messages() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(messages=[])
    assert graph.router(state) == END


def test_router_to_end_human_message() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(
        messages=[HumanMessage(content="hello")],
    )
    assert graph.router(state) == END


# ---------------------------------------------------------------------------
# DataFabricGraph — tool_router
# ---------------------------------------------------------------------------


def test_tool_router_success_ends() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(last_tool_success=True)
    assert graph.tool_router(state) == END


def test_tool_router_failure_retries() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(last_tool_success=False)
    assert graph.tool_router(state) == "inner_llm"


# ---------------------------------------------------------------------------
# DataFabricGraph — termination_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_termination_node_basic() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(iteration_count=5)
    result = await graph.termination_node(state)
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert "5 SQL attempts" in msg.content
    assert "rephrasing" in msg.content


@pytest.mark.asyncio
async def test_termination_node_with_error_info() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(
        iteration_count=3,
        last_error_category="bad_sql",
        last_error_detail="Invalid column 'foo'",
    )
    result = await graph.termination_node(state)
    msg = result["messages"][0]
    assert "bad_sql" in msg.content
    assert "Invalid column 'foo'" in msg.content


# ---------------------------------------------------------------------------
# DataFabricGraph — tool_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_node_no_tool_calls() -> None:
    graph = _make_graph()
    state = DataFabricSubgraphState(
        messages=[HumanMessage(content="hi")],
        iteration_count=2,
    )
    result = await graph.tool_node(state)
    assert result["iteration_count"] == 2
    assert "messages" not in result


@pytest.mark.asyncio
async def test_tool_node_success() -> None:
    graph = _make_graph()
    # Mock the execute_sql_tool to return a success result
    graph._execute_sql_tool = MagicMock()
    graph._execute_sql_tool.ainvoke = AsyncMock(
        return_value={"records": [{"id": 1}], "total_count": 1, "sql_query": "SELECT 1"}
    )

    state = DataFabricSubgraphState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "execute_sql",
                        "args": {"sql_query": "SELECT 1"},
                    }
                ],
            )
        ],
        iteration_count=0,
    )
    result = await graph.tool_node(state)
    assert result["last_tool_success"] is True
    assert result["iteration_count"] == 1
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)


@pytest.mark.asyncio
async def test_tool_node_failure_extracts_category() -> None:
    graph = _make_graph()
    graph._execute_sql_tool = MagicMock()
    graph._execute_sql_tool.ainvoke = AsyncMock(
        return_value={
            "records": [],
            "total_count": 0,
            "error": "[SQL_VALIDATION] (category: bad_sql) Invalid column",
            "sql_query": "SELECT bad",
        }
    )

    state = DataFabricSubgraphState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "execute_sql",
                        "args": {"sql_query": "SELECT bad"},
                    }
                ],
            )
        ],
    )
    result = await graph.tool_node(state)
    assert result["last_tool_success"] is False
    assert result["last_error_category"] == "bad_sql"
    assert "SQL_VALIDATION" in result["last_error_detail"]


@pytest.mark.asyncio
async def test_tool_node_value_error() -> None:
    graph = _make_graph()
    graph._execute_sql_tool = MagicMock()
    graph._execute_sql_tool.ainvoke = AsyncMock(side_effect=ValueError("bad input"))

    state = DataFabricSubgraphState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc1", "name": "execute_sql", "args": {"sql_query": "X"}}
                ],
            )
        ],
    )
    result = await graph.tool_node(state)
    assert result["last_tool_success"] is False
    assert "bad input" in result["last_error_detail"]


@pytest.mark.asyncio
async def test_tool_node_preserves_prior_error_on_no_new_error() -> None:
    graph = _make_graph()
    graph._execute_sql_tool = MagicMock()
    # Return empty records with no error — not a success (total_count=0) but no error string
    graph._execute_sql_tool.ainvoke = AsyncMock(
        return_value={"records": [], "total_count": 0, "sql_query": "SELECT 1"}
    )

    state = DataFabricSubgraphState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "execute_sql", "args": {}}],
            )
        ],
        last_error_category="prior_cat",
        last_error_detail="prior detail",
    )
    result = await graph.tool_node(state)
    # No new error, so prior values should be preserved
    assert result["last_error_category"] == "prior_cat"
    assert result["last_error_detail"] == "prior detail"


# ---------------------------------------------------------------------------
# DataFabricGraph.create
# ---------------------------------------------------------------------------


def test_create_returns_compiled_graph() -> None:
    llm = MagicMock(spec=["model_copy"])
    bound = MagicMock()
    copy = MagicMock()
    copy.bind_tools = MagicMock(return_value=bound)
    llm.model_copy.return_value = copy

    with patch(
        "uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph.datafabric_prompt_builder"
    ) as mock_pb:
        mock_pb.build.return_value = "prompt"
        compiled = DataFabricGraph.create(
            llm, [_make_entity("T")], _make_entities_service()
        )

    assert compiled is not None


# ---------------------------------------------------------------------------
# CATEGORY_MARKER extraction
# ---------------------------------------------------------------------------


def test_category_marker_extraction() -> None:
    error_str = "[SQL_VALIDATION] (category: bad_sql) Invalid column"
    start = error_str.index(CATEGORY_MARKER) + len(CATEGORY_MARKER)
    end = error_str.index(")", start)
    assert error_str[start:end] == "bad_sql"


def test_category_marker_not_present() -> None:
    error_str = "some generic error"
    assert CATEGORY_MARKER not in error_str


def test_category_marker_malformed_no_closing_paren() -> None:
    """Malformed error with marker but no closing ')' should not crash."""
    error_str = "[ERR] (category: bad_sql oops"
    assert CATEGORY_MARKER in error_str
    start = error_str.index(CATEGORY_MARKER) + len(CATEGORY_MARKER)
    end = error_str.find(")", start)
    assert end == -1  # no closing paren found


@pytest.mark.asyncio
async def test_tool_node_malformed_category_does_not_crash() -> None:
    """Tool node should not crash on malformed category marker."""
    graph = _make_graph()
    graph._execute_sql_tool = MagicMock()
    graph._execute_sql_tool.ainvoke = AsyncMock(
        return_value={
            "records": [],
            "total_count": 0,
            "error": "[ERR] (category: bad_sql oops no closing paren",
            "sql_query": "SELECT bad",
        }
    )

    state = DataFabricSubgraphState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "execute_sql",
                        "args": {"sql_query": "SELECT bad"},
                    }
                ],
            )
        ],
    )
    result = await graph.tool_node(state)
    assert result["last_tool_success"] is False
    # Category should be empty since parsing couldn't find closing ')'
    assert result["last_error_category"] == ""


# ---------------------------------------------------------------------------
# QueryExecutor — error path with OTEL span active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_executor_error_with_span_sets_attributes() -> None:
    """Error path when an OTEL span is active — verifies span attributes are set."""
    enriched = _make_enriched_exception("sql error")

    svc = _make_entities_service(error=enriched)
    qe = QueryExecutor(svc, [_make_entity("T")])

    fake_df_error = _FakeDataFabricError(
        code="SQL_VALIDATION",
        message="bad column",
        trace_id="trace-1",
        category=_FakeCategory.BAD_SQL,
        is_bad_sql=True,
    )

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
        return_value=False
    )

    with (
        patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
        patch(
            "uipath_langchain.agent.tools.datafabric_tool.datafabric_subgraph.DataFabricError"
        ) as mock_dfe_cls,
    ):
        mock_dfe_cls.from_enriched_exception.return_value = fake_df_error
        result = await qe("SELECT bad")

    assert result["records"] == []
    # Verify span attributes were set
    set_attr_calls = {
        call.args[0]: call.args[1] for call in mock_span.set_attribute.call_args_list
    }
    assert set_attr_calls["df.success"] is False
    assert set_attr_calls["df.error.code"] == "SQL_VALIDATION"
    assert set_attr_calls["df.error.message"] == "bad column"
    assert set_attr_calls["df.error.trace_id"] == "trace-1"
    assert set_attr_calls["df.error.category"] == "bad_sql"
    mock_span.record_exception.assert_called_once()
    mock_span.set_status.assert_called_once()


@pytest.mark.asyncio
async def test_query_executor_error_with_span_no_df_error() -> None:
    """Error path with OTEL span but a plain (non-EnrichedException) error."""
    svc = _make_entities_service(error=RuntimeError("timeout"))
    qe = QueryExecutor(svc, [_make_entity("T")])

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
        return_value=False
    )

    with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
        result = await qe("SELECT 1")

    assert result["error"] == "timeout"
    set_attr_calls = {
        call.args[0]: call.args[1] for call in mock_span.set_attribute.call_args_list
    }
    assert set_attr_calls["df.success"] is False
    assert "timeout" in set_attr_calls["df.error.raw"]
    # No df_error attributes should be set
    assert "df.error.code" not in set_attr_calls


# ---------------------------------------------------------------------------
# DataFabricGraph — llm_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_node_invokes_inner_llm() -> None:
    graph = _make_graph()
    mock_response = AIMessage(content="I'll query the database")
    graph._inner_llm = MagicMock()
    graph._inner_llm.ainvoke = AsyncMock(return_value=mock_response)

    state = DataFabricSubgraphState(messages=[HumanMessage(content="How many orders?")])
    result = await graph.llm_node(state)
    assert result["messages"] == [mock_response]
    # Verify system message was prepended
    call_args = graph._inner_llm.ainvoke.call_args[0][0]
    assert call_args[0] == graph._system_message
    assert call_args[1].content == "How many orders?"
