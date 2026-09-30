"""Tests for finishing a run whose model can't be forced and answered in text."""

from unittest.mock import Mock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from uipath_langchain.agent.react.no_forced_tool_choice import (
    finish_without_forcing,
    finishing_tool_call,
    output_fields,
)
from uipath_langchain.agent.react.tools import create_flow_control_tools
from uipath_langchain.chat.handlers.base import DefaultModelPayloadHandler


class _Output(BaseModel):
    answer: str
    confidence: float


class _NoFields(BaseModel):
    pass


def _search_tool() -> StructuredTool:
    return StructuredTool(
        name="search",
        description="search",
        args_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        func=lambda **_: None,
    )


class TestOutputFields:
    def test_are_the_end_execution_arguments(self) -> None:
        fields = output_fields([_search_tool(), *create_flow_control_tools(_Output)])
        assert fields == ["answer", "confidence"]

    def test_output_without_fields(self) -> None:
        assert output_fields(create_flow_control_tools(_NoFields)) == []

    def test_none_without_end_execution(self) -> None:
        assert output_fields([_search_tool()]) is None


class TestFinishWithoutForcing:
    def test_binds_only_end_execution_and_raise_error_on_auto(self) -> None:
        model = Mock(spec=BaseChatModel)
        tools = [_search_tool(), *create_flow_control_tools(_Output)]

        finish_without_forcing(
            model, DefaultModelPayloadHandler(model), tools, [], AIMessage("x")
        )

        bound, kwargs = model.bind_tools.call_args
        assert [tool.name for tool in bound[0]] == [
            END_EXECUTION_TOOL.name,
            RAISE_ERROR_TOOL.name,
        ]
        assert kwargs["tool_choice"] == "auto"

    def test_history_then_the_answer_then_the_request(self) -> None:
        model = Mock(spec=BaseChatModel)
        history = [HumanMessage("Which city is warmer?")]
        answer = AIMessage(content="Lisbon is warmer at 27 °C.")

        _, messages = finish_without_forcing(
            model,
            DefaultModelPayloadHandler(model),
            create_flow_control_tools(_Output),
            history,
            answer,
        )

        assert messages[:-1] == [*history, answer]
        assert isinstance(messages[-1], HumanMessage)
        assert END_EXECUTION_TOOL.name in messages[-1].text
        assert RAISE_ERROR_TOOL.name in messages[-1].text
        assert history == [HumanMessage("Which city is warmer?")]


_FIELDS = ["answer", "confidence"]


def _end_execution(args: dict[str, object]) -> AIMessage:
    return AIMessage(
        content=[{"type": "thinking", "thinking": "", "signature": "sig"}],
        tool_calls=[{"name": END_EXECUTION_TOOL.name, "args": args, "id": "toolu_1"}],
    )


class TestFinishingToolCall:
    def test_end_execution_call_is_kept_without_its_thinking(self) -> None:
        final = finishing_tool_call(
            _end_execution({"answer": "Lisbon", "confidence": 1.0}), _FIELDS
        )

        assert final is not None
        assert final.tool_calls[0]["id"] == "toolu_1"
        assert final.tool_calls[0]["args"] == {"answer": "Lisbon", "confidence": 1.0}
        assert final.content == []

    def test_empty_end_execution_call_gives_none(self) -> None:
        for empty in ({}, {"answer": None}, {"answer": "", "confidence": None}):
            assert finishing_tool_call(_end_execution(empty), _FIELDS) is None

    def test_zero_and_false_are_output(self) -> None:
        assert finishing_tool_call(_end_execution({"confidence": 0}), _FIELDS)
        assert finishing_tool_call(_end_execution({"answer": False}), _FIELDS)

    def test_output_without_fields_can_be_empty(self) -> None:
        final = finishing_tool_call(_end_execution({}), [])
        assert final is not None
        assert final.tool_calls[0]["args"] == {}

    def test_raise_error_call_is_kept(self) -> None:
        reply = AIMessage(
            content=[{"type": "thinking", "thinking": "", "signature": "sig"}],
            tool_calls=[
                {"name": RAISE_ERROR_TOOL.name, "args": {"message": "x"}, "id": "t2"}
            ],
        )

        final = finishing_tool_call(reply, _FIELDS)

        assert final is not None
        assert final.tool_calls[0]["name"] == RAISE_ERROR_TOOL.name
        assert final.content == []
