"""Advanced agents on models that reject a forced tool choice (Opus 5.5).

Not mocked: create_agent derives tool_choice from response_format after middleware
runs, so only a real graph shows what the model is bound with.
"""

import asyncio
from typing import Any

import pytest
from langchain.agents.structured_output import ToolStrategy
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from uipath_langchain.agent.advanced.agent import create_advanced_agent
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

OUTPUT_SCHEMA = {
    "title": "end_execution",
    "description": "Return the final output.",
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class _Model(GenericFakeChatModel):
    model_name: str = "claude-opus-5-5"
    model_details: dict[str, Any] = {}
    bindings: list[dict[str, Any]] = []
    calls: list[list[Any]] = []

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return {"ls_provider": "anthropic", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.bindings.append({"tools": list(tools), **kwargs})
        return self

    def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _model(script: list[AIMessage], *, skip_forced: bool) -> _Model:
    return _Model(
        messages=iter(script),
        model_details={"shouldSkipForcedToolChoice": True} if skip_forced else {},
        bindings=[],
        calls=[],
    )


def _end(answer: str, call_id: str = "end-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "end_execution", "args": {"answer": answer}, "id": call_id}
        ],
    )


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return query


def _run(model: BaseChatModel) -> dict[str, Any]:
    graph = create_advanced_agent(
        model=model, tools=[lookup], response_format=ToolStrategy(OUTPUT_SCHEMA)
    )
    return asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}))


def _tool_names(binding: dict[str, Any]) -> list[str]:
    names = []
    for bound in binding["tools"]:
        if isinstance(bound, dict):
            names.append(bound.get("function", bound).get("name"))
        else:
            names.append(bound.name)
    return names


class TestSkipForcedModel:
    def test_output_tool_is_bound_without_forcing(self) -> None:
        """Opus 5.5 answers a forced tool_choice with a 400."""
        model = _model([_end("42")], skip_forced=True)

        _run(model)

        forced = _model([_end("42")], skip_forced=False)
        _run(forced)

        assert model.bindings[0]["tool_choice"] == "auto"
        assert sorted(_tool_names(model.bindings[0])) == sorted(
            _tool_names(forced.bindings[0])
        )

    def test_output_tool_call_ends_the_run_with_the_structured_response(
        self,
    ) -> None:
        model = _model([_end("42")], skip_forced=True)

        result = _run(model)

        assert result["structured_response"] == {"answer": "42"}
        assert [b["tool_choice"] for b in model.bindings] == ["auto"]
        assert len(model.calls) == 1
        reply = result["messages"][-1]
        assert isinstance(reply, ToolMessage)
        assert reply.tool_call_id == "end-1"

    def test_a_text_turn_is_nudged_and_the_retry_is_not_forced_either(
        self,
    ) -> None:
        """The retry from _PayloadHandlerMiddleware must go through the same path."""
        model = _model(
            [AIMessage(content="The answer is 42."), _end("42")], skip_forced=True
        )

        result = _run(model)

        assert result["structured_response"] == {"answer": "42"}
        assert [b["tool_choice"] for b in model.bindings] == ["auto", "auto"]
        assert isinstance(model.calls[1][-1], HumanMessage)

    def test_two_text_turns_fail_the_run(self) -> None:
        model = _model(
            [AIMessage(content="thinking aloud"), AIMessage(content="still here")],
            skip_forced=True,
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            _run(model)

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.LLM_INVALID_RESPONSE
        )

    def test_two_output_calls_are_sent_back_as_errors_like_tool_strategy(
        self,
    ) -> None:
        both = AIMessage(
            content="",
            tool_calls=[
                {"name": "end_execution", "args": {"answer": "a"}, "id": "e1"},
                {"name": "end_execution", "args": {"answer": "b"}, "id": "e2"},
            ],
        )
        model = _model([both, _end("b", "e3")], skip_forced=True)

        result = _run(model)

        errors = [
            m
            for m in result["messages"]
            if isinstance(m, ToolMessage) and m.tool_call_id in ("e1", "e2")
        ]
        assert len(errors) == 2
        assert all("Please fix your mistakes" in m.content for m in errors)
        assert [b["tool_choice"] for b in model.bindings] == ["auto", "auto"]
        assert result["structured_response"] == {"answer": "b"}


class TestForcedModel:
    def test_other_models_stay_forced(self) -> None:
        model = _model([_end("42")], skip_forced=False)

        result = _run(model)

        assert model.bindings[0]["tool_choice"] == "any"
        assert result["structured_response"] == {"answer": "42"}


class _AnthropicModel(ChatAnthropic):
    """A real ChatAnthropic; records the request body instead of sending it."""

    model_details: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.payloads.append(self._get_request_payload(messages, stop=stop, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=_end("42"))])

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)


def test_the_anthropic_request_is_unforced_and_keeps_the_tool_cache_breakpoint() -> (
    None
):
    """The prompt caching middleware only tags the last tool if it is a BaseTool."""
    model = _AnthropicModel(
        model="claude-opus-5-5",
        api_key="unused",
        model_details={"shouldSkipForcedToolChoice": True},
        payloads=[],
    )

    _run(model)

    payload = model.payloads[0]
    assert payload["tool_choice"] == {"type": "auto"}
    assert "end_execution" in [t["name"] for t in payload["tools"]]
    assert any("cache_control" in t for t in payload["tools"])
