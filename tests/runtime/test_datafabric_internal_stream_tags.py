"""Tests for keeping a tool's inner sub-graph messages out of the conversation.

The Data Fabric tool runs its own inner LangGraph (an inner LLM plus the
inner-only ``execute_sql`` tool). When the outer graph is streamed with
``stream_mode="messages", subgraphs=True`` those inner messages surface too, and
without suppression they leak into the conversation transcript. Replayed as
history the next turn, they make the outer router treat ``execute_sql`` as a
tool the agent called and route to it — ``AGENT_RUNTIME.ROUTING_ERROR``.

The fix tags the nested ``ainvoke`` with LangGraph's own streaming-suppression
tags. Two are required because inner messages reach the ``messages`` stream via
two different paths in ``StreamMessagesHandler``:

* ``nostream`` gates ``on_chat_model_start`` — suppresses the inner **LLM**
  ``AIMessage`` (the ``execute_sql`` tool call).
* ``langsmith:hidden`` gates ``on_chain_start`` — suppresses the node-output
  path, i.e. the inner **ToolMessage** (the ``execute_sql`` result).

Neither alone is sufficient (see ``test_single_tag_is_insufficient``); together
they suppress the whole pair symmetrically. The OTEL tool-call spans are created
independently of these tags, so the inner steps stay visible in the trace UI.

These tests build a graph with the *same shape* as the Data Fabric tool and run
it through real LangGraph streaming, so the tag choice is verified against the
actual streaming contract rather than assumed.
"""

from typing import Annotated, Any, List, Optional, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# The tags the production code applies. Kept as a literal here so these tests are
# self-contained (importing the tool pulls in the whole tools package). The
# ``test_production_tags_match`` case asserts the production constant equals this.
EXPECTED_TAGS = ["nostream", "langsmith:hidden"]


class _State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


@tool
def execute_sql(sql_query: str) -> str:
    """Inner-only tool that must never leak to the outer conversation."""
    return "rows: [{'count': 1}]"


def _build_outer_graph(inner_tags: Optional[List[str]]) -> Any:
    """Outer agent + an ``Entities`` tool whose body runs an inner sub-graph.

    ``inner_tags`` are applied to the nested ``ainvoke`` (``None`` = untagged,
    the leak-reproducing control).
    """
    inner_llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_sql",
                        "args": {"sql_query": "SELECT 1"},
                        "id": "sql-1",
                    }
                ],
            ),
            AIMessage(content="inner final answer"),
        ]
    )

    def inner_llm_node(state: _State) -> dict[str, Any]:
        return {"messages": [inner_llm.invoke(state["messages"])]}

    async def inner_tool_node(state: _State) -> dict[str, Any]:
        result = await execute_sql.ainvoke({"sql_query": "SELECT 1"})
        return {"messages": [ToolMessage(content=result, tool_call_id="sql-1")]}

    def inner_route(state: _State) -> str:
        last = state["messages"][-1]
        return "itool" if getattr(last, "tool_calls", None) else END

    inner = StateGraph(_State)
    inner.add_node("illm", inner_llm_node)
    inner.add_node("itool", inner_tool_node)
    inner.add_edge(START, "illm")
    inner.add_conditional_edges("illm", inner_route, {"itool": "itool", END: END})
    inner.add_edge("itool", "illm")
    inner_graph = inner.compile()

    async def entities_body(state: _State) -> dict[str, Any]:
        config = {"tags": inner_tags} if inner_tags is not None else None
        await inner_graph.ainvoke(
            {"messages": [HumanMessage(content="inner q")]}, config
        )
        return {"messages": [ToolMessage(content="entities answer", tool_call_id="e1")]}

    entities = StateGraph(_State)
    entities.add_node("entities_body", entities_body)
    entities.add_edge(START, "entities_body")
    entities.add_edge("entities_body", END)
    entities_sub = entities.compile()

    outer_llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "Entities", "args": {"user_query": "q"}, "id": "e1"}
                ],
            ),
            AIMessage(content="outer final answer"),
        ]
    )

    def agent_body(state: _State) -> dict[str, Any]:
        return {"messages": [outer_llm.invoke(state["messages"])]}

    agent = StateGraph(_State)
    agent.add_node("agent_body", agent_body)
    agent.add_edge(START, "agent_body")
    agent.add_edge("agent_body", END)
    agent_sub = agent.compile()

    def route(state: _State) -> str:
        last = state["messages"][-1]
        return "Entities" if getattr(last, "tool_calls", None) else END

    outer = StateGraph(_State)
    outer.add_node("agent", agent_sub)
    outer.add_node("Entities", entities_sub)
    outer.add_edge(START, "agent")
    outer.add_conditional_edges("agent", route, {"Entities": "Entities", END: END})
    outer.add_edge("Entities", "agent")
    return outer.compile()


async def _streamed_inner_messages(inner_tags: Optional[List[str]]) -> List[Any]:
    """Return the inner sub-graph messages that surface in the ``messages`` stream."""
    graph = _build_outer_graph(inner_tags)
    inner: List[Any] = []
    async for ns, (msg, _meta) in graph.astream(
        {"messages": [HumanMessage("go")]},
        stream_mode="messages",
        subgraphs=True,
    ):
        ns_str = str(ns)
        if "entities_body" in ns_str or "illm" in ns_str or "itool" in ns_str:
            inner.append(msg)
    return inner


def _tool_call_names(messages: List[Any]) -> set[str]:
    names: set[str] = set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            names.add(tc["name"])
    return names


async def test_internal_tags_suppress_inner_messages() -> None:
    """With the production tags, no inner sub-graph message reaches the stream."""
    inner = await _streamed_inner_messages(EXPECTED_TAGS)

    assert "execute_sql" not in _tool_call_names(inner), (
        "inner execute_sql tool call leaked into the conversation stream"
    )
    assert not any(getattr(m, "tool_call_id", None) == "sql-1" for m in inner), (
        "inner execute_sql ToolMessage leaked into the conversation stream"
    )
    assert inner == [], f"unexpected inner messages leaked: {inner}"


async def test_control_without_tags_reproduces_the_leak() -> None:
    """Untagged, the inner execute_sql call leaks — proving the test bites."""
    inner = await _streamed_inner_messages(None)
    assert "execute_sql" in _tool_call_names(inner)


async def test_single_tag_is_insufficient() -> None:
    """Each tag alone leaves one of the two emission paths open.

    ``nostream`` alone leaves node-output messages; ``langsmith:hidden`` alone
    leaves the inner LLM message. This is why the fix uses both.
    """
    only_nostream = await _streamed_inner_messages(["nostream"])
    only_hidden = await _streamed_inner_messages(["langsmith:hidden"])

    assert only_nostream != [], "nostream alone unexpectedly suppressed everything"
    assert only_hidden != [], (
        "langsmith:hidden alone unexpectedly suppressed everything"
    )


def test_production_tags_match() -> None:
    """The tool applies exactly the tags these tests validate.

    Skipped only when the tools package can't be imported in the local
    environment (e.g. an optional dependency like ``a2a`` is missing); it runs
    normally in CI, catching any drift in the production tag list.
    """
    try:
        from uipath_langchain.agent.tools.datafabric_tool.datafabric_tool import (
            _INTERNAL_SUBGRAPH_TAGS,
        )
    except Exception as exc:  # pragma: no cover - env-dependent import guard
        pytest.skip(f"tools package not importable in this environment: {exc}")

    assert _INTERNAL_SUBGRAPH_TAGS == EXPECTED_TAGS
