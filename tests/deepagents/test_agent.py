from unittest.mock import MagicMock, patch

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from uipath_langchain.deepagents import create_uipath_deep_agent
from uipath_langchain.deepagents.backend import _UiPathWorkspaceBackendFactory
from uipath_langchain.deepagents.metadata import requires_managed_workspace


class _State(TypedDict):
    value: str


def _compiled_graph():
    builder = StateGraph(_State)
    builder.add_node("noop", lambda state: state)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile()


def test_api_forwards_upstream_contract_with_uipath_backend() -> None:
    compiled = _compiled_graph()
    model = MagicMock()
    tool = MagicMock()
    middleware = MagicMock()
    subagent = MagicMock()
    response_format = MagicMock()
    context_schema = MagicMock()

    with patch(
        "uipath_langchain.deepagents.agent._create_deep_agent",
        return_value=compiled,
    ) as upstream:
        graph = create_uipath_deep_agent(
            model=model,
            tools=[tool],
            system_prompt="system",
            middleware=[middleware],
            subagents=[subagent],
            skills=["/skills/"],
            memory=["/memory/AGENTS.md"],
            permissions=[],
            response_format=response_format,
            context_schema=context_schema,
        )

    kwargs = upstream.call_args.kwargs
    assert kwargs["model"] is model
    assert kwargs["tools"] == [tool]
    assert kwargs["system_prompt"] == "system"
    assert kwargs["middleware"] == [middleware]
    assert kwargs["subagents"] == [subagent]
    assert kwargs["skills"] == ["/skills/"]
    assert kwargs["memory"] == ["/memory/AGENTS.md"]
    assert kwargs["permissions"] == []
    assert kwargs["response_format"] is response_format
    assert kwargs["context_schema"] is context_schema
    assert "checkpointer" not in kwargs
    assert "store" not in kwargs
    assert "cache" not in kwargs
    assert "debug" not in kwargs
    assert "name" not in kwargs
    assert "interrupt_on" not in kwargs
    assert isinstance(kwargs["backend"], _UiPathWorkspaceBackendFactory)
    assert requires_managed_workspace(graph)
