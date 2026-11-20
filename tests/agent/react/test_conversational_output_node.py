"""Tests for the GENERATE_CONVERSATIONAL_OUTPUT node."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import TAG_NOSTREAM
from pydantic import BaseModel, Field
from uipath.agent.react import (
    SET_CONVERSATIONAL_OUTPUT_TOOL,
    get_chat_system_prompt,
    get_generate_output_prompt,
)

from uipath_langchain.agent.react.conversational_output_node import (
    create_conversational_output_extractor,
    create_conversational_output_node,
)
from uipath_langchain.agent.react.types import AgentGraphState, InnerAgentGraphState
from uipath_langchain.agent.react.utils import (
    build_conversational_output_args_schema,
)


class _OutputSchema(BaseModel):
    uipath__agent_response_messages: list[Any] = Field(default_factory=list)
    handoff_target: str = "none"
    ready_for_handoff: bool = False


def _make_state(messages: list[Any]) -> AgentGraphState:
    return AgentGraphState(
        messages=messages,
        inner_state=InnerAgentGraphState(initial_message_count=1),
    )


def _make_mock_model(
    *, skip_forced: bool = False
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return a chat model whose
    `model_copy(...).bind_tools(...).ainvoke(...)` chain records the messages
    and returns a canned AIMessage with the set_conversational_output tool
    call.

    The TAG_NOSTREAM tag is supplied at call time via the `config=` kwarg
    (mirroring the analyze-files-tool pattern), so `bind_tools` returns the
    runnable that's invoked directly.

    `skip_forced` sets the discovery flag marking a model that rejects a forced
    tool choice. It is set explicitly either way: on a bare MagicMock the
    lookup returns a truthy mock and silently reads as true.

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
    model.model_details = {"shouldSkipForcedToolChoice": True} if skip_forced else {}
    model.model_copy = MagicMock(return_value=non_streaming)
    return model, non_streaming, bound


class TestCreateConversationalOutputNode:
    @pytest.mark.asyncio
    async def test_returns_extracted_args_via_inner_state(self):
        model, _non_streaming, _bound = _make_mock_model()
        node = create_conversational_output_node(model, _OutputSchema)

        state = _make_state([SystemMessage(content="sys"), HumanMessage(content="hi")])
        result = await node(state)

        assert "messages" not in result
        assert result["inner_state"] == {
            "conversational_output": {
                "handoff_target": "billing",
                "ready_for_handoff": True,
            }
        }

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

        # Nothing is written to the messages channel — args flow via inner_state.
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_stripping_does_not_write_back_into_graph_state(self):
        """The node hands `state.messages` to the extractor directly, so a
        strip that edited the system message in place would leave graph state
        holding a prompt stripped of its output-fields section -- the loop
        would silently lose it. Only the call's copy may be rewritten.
        """
        model, _non_streaming, bound = _make_mock_model()
        node = create_conversational_output_node(model, _OutputSchema)
        section_bearing = get_chat_system_prompt(
            model="claude-3-sonnet",
            system_message="builder prompt",
            agent_name="Agent",
            output_schema=build_conversational_output_args_schema(_OutputSchema),
        )
        state = _make_state(
            [SystemMessage(content=section_bearing), HumanMessage(content="hi")]
        )

        await node(state)

        sent = bound.ainvoke.call_args.args[0]
        assert "STRUCTURED OUTPUT FOR THIS TURN" not in sent[0].content
        assert "builder prompt" in sent[0].content
        assert state.messages[0].content == section_bearing

    @pytest.mark.asyncio
    async def test_binds_only_set_conversational_output_tool(self):
        model, non_streaming, _bound = _make_mock_model()
        create_conversational_output_node(model, _OutputSchema)

        non_streaming.bind_tools.assert_called_once()
        tools_arg = non_streaming.bind_tools.call_args.args[0]
        assert len(tools_arg) == 1
        assert tools_arg[0].name == SET_CONVERSATIONAL_OUTPUT_TOOL.name

    @pytest.mark.asyncio
    async def test_forces_tool_choice_by_default(self):
        model, non_streaming, _bound = _make_mock_model()
        create_conversational_output_node(model, _OutputSchema)

        tool_choice = non_streaming.bind_tools.call_args.kwargs.get("tool_choice")
        assert tool_choice == "any"

    @pytest.mark.asyncio
    async def test_unforces_tool_choice_when_the_model_rejects_forcing(self):
        """Opus 5.5 and Sonnet 5.5 answer a forced tool choice with a 400, so
        discovery flags them and the extraction call goes out on "auto"."""
        model, non_streaming, _bound = _make_mock_model(skip_forced=True)
        create_conversational_output_node(model, _OutputSchema)

        tool_choice = non_streaming.bind_tools.call_args.kwargs.get("tool_choice")
        assert tool_choice == "auto"

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

    @pytest.mark.asyncio
    async def test_raises_when_llm_omits_set_conversational_output_call(self):
        """When the internal LLM returns an AIMessage without the expected
        set_conversational_output tool call, the node raises a structured
        AgentRuntimeError rather than propagating an empty inner_state
        payload downstream."""
        from uipath_langchain.agent.exceptions import (
            AgentRuntimeError,
            AgentRuntimeErrorCode,
        )

        response = AIMessage(content="no tool call here", tool_calls=[])
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=response)
        non_streaming = MagicMock()
        non_streaming.bind_tools = MagicMock(return_value=bound)
        model = MagicMock()
        model.model_copy = MagicMock(return_value=non_streaming)

        node = create_conversational_output_node(model, _OutputSchema)
        state = _make_state([SystemMessage(content="sys"), HumanMessage(content="hi")])

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(state)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.LLM_INVALID_RESPONSE
        )


class TestExtractionSeesTheStoredMessages:
    """The extraction call replays state verbatim.

    The output-fields section is added by the loop's model call and never
    stored, so there is nothing here to strip back out.
    """

    @pytest.mark.asyncio
    async def test_system_message_is_passed_through_untouched(self):
        model, _non_streaming, bound = _make_mock_model()
        extract = create_conversational_output_extractor(model, _OutputSchema)
        system = SystemMessage(
            content=get_chat_system_prompt(
                model="claude-3-sonnet",
                system_message="Set ready_for_handoff once billing is confirmed.",
                agent_name="Agent",
            )
        )
        human = HumanMessage(content="hi")

        await extract([system, human])

        sent = bound.ainvoke.call_args.args[0]
        assert sent[0] is system
        assert sent[1] is human
        assert "STRUCTURED OUTPUT FOR THIS TURN" not in sent[0].content
        assert "Set ready_for_handoff once billing is confirmed." in sent[0].content

    @pytest.mark.asyncio
    async def test_generate_output_prompt_still_trails_the_messages(self):
        model, _non_streaming, bound = _make_mock_model()
        extract = create_conversational_output_extractor(model, _OutputSchema)

        await extract([SystemMessage(content="sys"), HumanMessage(content="hi")])

        sent = bound.ainvoke.call_args.args[0]
        assert sent[-1].content == get_generate_output_prompt()


class TestOnlyTheFirstMessageIsConsidered:
    """The section can only be in the system message at index 0.

    The standard loop's init node puts the freshly built prompt there and drops
    any stale one from the replayed history; the advanced wrapper prepends its
    own prompt at index 0 before calling the extractor.
    """

    @pytest.mark.asyncio
    async def test_a_later_system_message_is_left_alone(self):
        model, _non_streaming, bound = _make_mock_model()
        extract = create_conversational_output_extractor(model, _OutputSchema)
        section_bearing = get_chat_system_prompt(
            model="claude-3-sonnet",
            system_message="builder prompt",
            agent_name="Agent",
            output_schema=build_conversational_output_args_schema(_OutputSchema),
        )
        stray = SystemMessage(content=section_bearing)

        await extract([SystemMessage(content="real prompt"), HumanMessage("hi"), stray])

        sent = bound.ainvoke.call_args.args[0]
        assert sent[0].content == "real prompt"
        assert sent[2] is stray

    @pytest.mark.asyncio
    async def test_a_history_with_no_system_message_is_tolerated(self):
        """A caller may hand over a bare transcript -- an advanced agent whose
        system prompt resolved to nothing, say. The rewrite must not apply."""
        model, _non_streaming, bound = _make_mock_model()
        extract = create_conversational_output_extractor(model, _OutputSchema)
        history = [HumanMessage("hi"), AIMessage(content="done")]

        await extract(history)

        sent = bound.ainvoke.call_args.args[0]
        assert sent[0] is history[0]
        assert sent[1] is history[1]

    @pytest.mark.asyncio
    async def test_empty_history_is_tolerated(self):
        model, _non_streaming, bound = _make_mock_model()
        extract = create_conversational_output_extractor(model, _OutputSchema)

        await extract([])

        assert len(bound.ainvoke.call_args.args[0]) == 1
