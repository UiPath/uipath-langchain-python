"""Tests for the GENERATE_CONVERSATIONAL_OUTPUT node."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import TAG_NOSTREAM
from pydantic import BaseModel, Field
from uipath.agent.react import SET_CONVERSATIONAL_OUTPUT_TOOL

from uipath_langchain.agent.react.conversational_output_node import (
    create_conversational_output_node,
)
from uipath_langchain.agent.react.types import AgentGraphState, InnerAgentGraphState


class _OutputSchema(BaseModel):
    uipath__agent_response_messages: list = Field(default_factory=list)
    handoff_target: str = "none"
    ready_for_handoff: bool = False


def _make_state(messages: list[Any]) -> AgentGraphState:
    return AgentGraphState(
        messages=messages,
        inner_state=InnerAgentGraphState(initial_message_count=1),
    )


def _make_mock_model() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return a chat model whose
    `model_copy(...).bind_tools(...).ainvoke(...)` chain records the messages
    and returns a canned AIMessage with the set_conversational_output tool
    call.

    The TAG_NOSTREAM tag is supplied at call time via the `config=` kwarg
    (mirroring the analyze-files-tool pattern), so `bind_tools` returns the
    runnable that's invoked directly.

    Returns (model, non_streaming, bound).
    """
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": SET_CONVERSATIONAL_OUTPUT_TOOL.name,
                "args": {"handoff_target": "billing", "ready_for_handoff": True},
                "id": "call_1",
            }
        ],
    )

    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=response)

    non_streaming = MagicMock()
    non_streaming.bind_tools = MagicMock(return_value=bound)

    model = MagicMock()
    model.model_copy = MagicMock(return_value=non_streaming)
    return model, non_streaming, bound


class TestCreateConversationalOutputNode:
    @pytest.mark.asyncio
    async def test_returns_response_with_tool_call(self):
        model, _non_streaming, _bound = _make_mock_model()
        node = create_conversational_output_node(model, _OutputSchema)

        state = _make_state([SystemMessage(content="sys"), HumanMessage(content="hi")])
        result = await node(state)

        assert len(result["messages"]) == 1
        ai = result["messages"][0]
        assert isinstance(ai, AIMessage)
        assert ai.tool_calls[0]["name"] == SET_CONVERSATIONAL_OUTPUT_TOOL.name
        assert ai.tool_calls[0]["args"]["handoff_target"] == "billing"

    @pytest.mark.asyncio
    async def test_appends_human_instruction_for_llm_call(self):
        """The framework instruction is appended as a HumanMessage for the LLM
        call, but never returned to state."""
        model, _non_streaming, bound = _make_mock_model()
        node = create_conversational_output_node(model, _OutputSchema)

        agent_reply = AIMessage(content="here's my reply")
        state = _make_state(
            [
                SystemMessage(content="sys"),
                HumanMessage(content="hi"),
                agent_reply,
            ]
        )
        result = await node(state)

        # The LLM was invoked with state.messages PLUS one extra HumanMessage.
        bound.ainvoke.assert_awaited_once()
        invoked_messages = bound.ainvoke.await_args.args[0]
        assert len(invoked_messages) == len(state.messages) + 1
        assert isinstance(invoked_messages[-1], HumanMessage)
        assert "set_conversational_output" in invoked_messages[-1].content

        # The instruction was NOT persisted into the returned messages.
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_binds_only_set_conversational_output_tool(self):
        model, non_streaming, _bound = _make_mock_model()
        create_conversational_output_node(model, _OutputSchema)

        non_streaming.bind_tools.assert_called_once()
        tools_arg = non_streaming.bind_tools.call_args.args[0]
        assert len(tools_arg) == 1
        assert tools_arg[0].name == SET_CONVERSATIONAL_OUTPUT_TOOL.name

    @pytest.mark.asyncio
    async def test_ainvoke_config_includes_tag_nostream(self):
        """TAG_NOSTREAM is added to the per-call config (mirroring the
        analyze-files-tool pattern) — not via .with_config at construction."""
        model, _non_streaming, bound = _make_mock_model()
        node = create_conversational_output_node(model, _OutputSchema)

        state = _make_state([SystemMessage(content="sys"), HumanMessage(content="hi")])
        await node(state)

        bound.ainvoke.assert_awaited_once()
        config_arg = bound.ainvoke.await_args.kwargs.get("config")
        assert config_arg is not None
        assert TAG_NOSTREAM in config_arg.get("tags", [])

    @pytest.mark.asyncio
    async def test_disables_streaming_on_internal_llm(self):
        """The node copies the model with `disable_streaming=True` so the
        underlying provider call doesn't stream — same pattern used by
        analyze-files."""
        model, _non_streaming, _bound = _make_mock_model()
        create_conversational_output_node(model, _OutputSchema)

        model.model_copy.assert_called_once()
        update_arg = model.model_copy.call_args.kwargs.get("update")
        assert update_arg is not None
        assert update_arg.get("disable_streaming") is True
