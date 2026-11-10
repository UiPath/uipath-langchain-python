import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tests.settings import agent_hub_dummy_settings
from uipath_langchain.chat.chat_model_factory import get_chat_model
from uipath_langchain.chat.handlers.anthropic import AnthropicPayloadHandler

_CLAUDE = "claude-opus-5"
_MARK = {"type": "ephemeral"}


def vertex_entry(subscription: str) -> dict[str, Any]:
    return {
        "modelName": _CLAUDE,
        "vendor": "VertexAi",
        "modelFamily": "AnthropicClaude",
        "modelSubscriptionType": subscription,
        "byomDetails": None,
        "modelDetails": {},
    }


@contextmanager
def discovered(entry: dict[str, Any]) -> Iterator[None]:
    with (
        patch.dict(os.environ, {"UIPATH_LLM_SERVICE": "agenthub"}),
        patch.object(
            type(agent_hub_dummy_settings), "get_available_models", return_value=[entry]
        ),
    ):
        yield


def vertex_claude() -> ChatAnthropic:
    model = get_chat_model(_CLAUDE, client_settings=agent_hub_dummy_settings)
    assert isinstance(model, ChatAnthropic)
    return model


def sent_payload(
    messages: list[AnyMessage], subscription: str = "UiPathOwned"
) -> dict[str, Any]:
    with discovered(vertex_entry(subscription)):
        model = vertex_claude()
        marked, kwargs = AnthropicPayloadHandler(model).apply_prompt_caching(messages)
        assert kwargs == {}
        return model._get_request_payload(marked)


def marker_count(payload: dict[str, Any]) -> int:
    return json.dumps(payload).count('"cache_control"')


def tool_turn() -> list[AnyMessage]:
    return [
        SystemMessage(content="You are an agent."),
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "c1"}]),
        ToolMessage(content="result", tool_call_id="c1"),
    ]


class TestAnthropicPromptCaching:
    def test_marks_the_system_prompt_and_the_last_message(self) -> None:
        payload = sent_payload(
            [SystemMessage(content="sys"), HumanMessage(content="q")]
        )

        assert payload["system"][-1]["cache_control"] == _MARK
        assert payload["messages"][-1]["content"][-1]["cache_control"] == _MARK
        assert "cache_control" not in payload
        assert marker_count(payload) == 2

    def test_marks_a_trailing_tool_result(self) -> None:
        payload = sent_payload(tool_turn())

        last_block = payload["messages"][-1]["content"][-1]
        assert last_block["type"] == "tool_result"
        assert last_block["cache_control"] == _MARK
        assert marker_count(payload) == 2

    def test_marks_the_last_block_of_a_multimodal_message(self) -> None:
        image = {"type": "image", "url": "https://example.com/receipt.png"}
        messages: list[AnyMessage] = [
            SystemMessage(content="sys"),
            HumanMessage(content=[{"type": "text", "text": "What is this?"}, image]),
        ]
        before = [m.model_dump() for m in messages]

        payload = sent_payload(messages)

        last = payload["messages"][-1]["content"]
        assert last[-1]["cache_control"] == _MARK
        assert "cache_control" not in last[0]
        assert [m.model_dump() for m in messages] == before

    def test_leaves_a_blank_system_prompt_alone(self) -> None:
        payload = sent_payload(
            [SystemMessage(content="\n\n"), HumanMessage(content="q")]
        )

        assert marker_count(payload) == 1
        assert payload["messages"][-1]["content"][-1]["cache_control"] == _MARK

    @pytest.mark.parametrize("subscription", ["BYOMAdded", "BYOMReplacedLikeForLike"])
    def test_byo_models_get_no_markers(self, subscription: str) -> None:
        payload = sent_payload(tool_turn(), subscription=subscription)

        assert marker_count(payload) == 0

    def test_failed_discovery_lookup_gets_no_markers(self) -> None:
        with discovered(vertex_entry("UiPathOwned")):
            model = vertex_claude()
        with patch.object(
            type(agent_hub_dummy_settings),
            "get_available_models",
            side_effect=RuntimeError("discovery unavailable"),
        ):
            marked, _ = AnthropicPayloadHandler(model).apply_prompt_caching(tool_turn())

        assert marker_count(model._get_request_payload(marked)) == 0

    def test_does_not_change_the_messages_it_was_given(self) -> None:
        messages = tool_turn()
        before = [m.model_dump() for m in messages]

        sent_payload(messages)

        assert [m.model_dump() for m in messages] == before
