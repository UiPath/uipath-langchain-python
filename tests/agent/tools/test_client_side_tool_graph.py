"""Client-side tool declarations reach the tool when it runs in a graph.

The exchange's declarations arrive as the `uipath__client_side_tools` input. They
must apply in the tool node: LangGraph runs each node in a copy of the context, so
a filter set in an earlier node (the init node) never reached the tool, and every
client-side tool stayed available whatever the client declared.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from uipath.agent.models.agent import AgentClientSideToolResourceConfig

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.client_side_tool import (
    UIPATH_CLIENT_SIDE_TOOLS_INPUT_KEY,
    apply_tool_filter,
    create_client_side_tool,
)
from uipath_langchain.agent.tools.tool_node import UiPathToolNode
from uipath_langchain.chat.hitl import REQUIRE_CONVERSATIONAL_CONFIRMATION


class _State(AgentGraphState):
    uipath__client_side_tools: list[str] | None = None


def _resource(name: str) -> AgentClientSideToolResourceConfig:
    return AgentClientSideToolResourceConfig(
        name=name,
        description="A client-side test tool",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        output_schema=None,
    )


async def _run(
    tool_name: str,
    declared: list[str] | None,
    *,
    requires_confirmation: bool = False,
    confirmation: MagicMock | None = None,
) -> ToolMessage:
    """Run an init-like node, then the client-side tool's node, and return the tool's message.

    The client's answer is stubbed: the interrupt returns the tool's input at once.
    `confirmation` stands in for the confirmation request, so a test can see whether
    the node asked for one.
    """
    tool = create_client_side_tool(_resource(tool_name))
    if requires_confirmation:
        tool.metadata = {
            **(tool.metadata or {}),
            REQUIRE_CONVERSATIONAL_CONFIRMATION: True,
        }

    def init(state: _State) -> dict[str, Any]:
        # What the init node used to do: set the filter in its own node. It must
        # not be what decides availability.
        declared_here = getattr(state, UIPATH_CLIENT_SIDE_TOOLS_INPUT_KEY)
        if declared_here is not None:
            apply_tool_filter(
                declared_here,
                {tool_name: {"input_schema": None, "output_schema": None}},
            )
        return {}

    graph = StateGraph(_State)
    graph.add_node("init", init)
    graph.add_node("tool", UiPathToolNode(tool))
    graph.add_edge(START, "init")
    graph.add_edge("init", "tool")
    graph.add_edge("tool", END)

    inputs = _State(
        messages=[
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": {"q": "x"}, "id": "tc-1"}],
            ),
        ],
        uipath__client_side_tools=declared,
    )
    with (
        patch(
            "uipath_langchain.agent.tools.client_side_tool.durable_interrupt",
            side_effect=lambda fn: fn,
        ),
        patch(
            "uipath_langchain.agent.tools.client_side_tool.mockable",
            side_effect=lambda **kw: lambda fn: fn,
        ),
        patch(
            "uipath_langchain.agent.tools.tool_node.request_conversational_tool_confirmation",
            confirmation or MagicMock(return_value=None),
        ),
    ):
        out = await graph.compile().ainvoke(inputs)
    message = out["messages"][-1]
    assert isinstance(message, ToolMessage)
    return message


async def test_undeclared_tool_is_not_available() -> None:
    message = await _run("show_map", declared=["get_weather"])
    assert message.status == "error"
    assert "not available" in str(message.content)


async def test_empty_declaration_makes_no_tool_available() -> None:
    message = await _run("show_map", declared=[])
    assert message.status == "error"
    assert "not available" in str(message.content)


async def test_declared_tool_runs() -> None:
    message = await _run("show_map", declared=["show_map"])
    assert message.status == "success"


async def test_no_declaration_keeps_every_tool_available() -> None:
    message = await _run("show_map", declared=None)
    assert message.status == "success"


async def test_undeclared_tool_never_asks_for_confirmation() -> None:
    """An undeclared tool that requires confirmation must not interrupt the run for it."""
    confirmation = MagicMock(return_value=None)
    message = await _run(
        "show_map",
        declared=[],
        requires_confirmation=True,
        confirmation=confirmation,
    )
    assert message.status == "error"
    assert "not available" in str(message.content)
    confirmation.assert_not_called()


async def test_declared_tool_still_asks_for_confirmation() -> None:
    confirmation = MagicMock(return_value=None)
    message = await _run(
        "show_map",
        declared=["show_map"],
        requires_confirmation=True,
        confirmation=confirmation,
    )
    assert message.status == "success"
    confirmation.assert_called_once()
