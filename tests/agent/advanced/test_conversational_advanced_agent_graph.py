"""Tests for the conversational advanced agent wrapper builder."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from uipath_langchain.agent.advanced.agent import (
    create_conversational_advanced_agent_graph,
)
from uipath_langchain.agent.advanced.types import (
    ConversationalAdvancedAgentGraphState,
)


def _mock_model() -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    return model


def _fake_inner_agent() -> Any:
    """A stand-in deepagent that appends one AI message."""

    def respond(state: ConversationalAdvancedAgentGraphState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="here is my plan", id="ai-1")]}

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(
        ConversationalAdvancedAgentGraphState
    )
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def test_wrapper_graph_has_conversational_nodes() -> None:
    graph = create_conversational_advanced_agent_graph(
        model=_mock_model(), tools=[], system_prompt="sys", backend=None
    )
    assert {
        "capture_exchange_start",
        "advanced_agent",
        "transform_output",
    } <= set(graph.nodes)


@pytest.mark.asyncio
async def test_outputs_only_new_messages_as_response_messages() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=_fake_inner_agent(),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(), tools=[], system_prompt="sys", backend=None
        ).compile()

    history = [
        HumanMessage(content="hi", id="u1"),
        AIMessage(content="hello", id="a1"),
        HumanMessage(content="make a plan", id="u2"),
    ]
    result = await graph.ainvoke({"messages": history})

    response_messages = result["uipath__agent_response_messages"]
    assert len(response_messages) == 1
    assert response_messages[0].role == "assistant"
    assert response_messages[0].content_parts[0].data.inline == "here is my plan"


@pytest.mark.asyncio
async def test_empty_history_still_produces_response() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent.create_advanced_agent",
        return_value=_fake_inner_agent(),
    ):
        graph = create_conversational_advanced_agent_graph(
            model=_mock_model(), tools=[], system_prompt="sys", backend=None
        ).compile()

    result = await graph.ainvoke({"messages": [HumanMessage(content="hi", id="u1")]})

    assert len(result["uipath__agent_response_messages"]) == 1
