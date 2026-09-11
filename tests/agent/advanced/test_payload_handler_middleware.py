"""Tests for the payload-handler middleware on the advanced agent."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from deepagents import create_deep_agent
from deepagents.middleware import SubAgentMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.advanced.agent import (
    _PayloadHandlerMiddleware,
    _subagents_with_middleware,
    create_advanced_agent,
)
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.chat.exceptions import ChatModelError

VALIDATED_TOOL_CONFIG = {"function_calling_config": {"mode": "VALIDATED"}}


@tool
def echo(text: str) -> str:
    """Echo the given text."""
    return text


def _gemini() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="dummy")


def _request(
    model: Any, tool_choice: Any = None, response_format: Any = None
) -> ModelRequest[Any]:
    return ModelRequest(
        model=model,
        messages=[],
        tools=[echo],
        tool_choice=tool_choice,
        response_format=response_format,
    )


def _response(*messages: AIMessage, structured: Any = None) -> ModelResponse[Any]:
    return ModelResponse(result=list(messages), structured_response=structured)


def _specs(subagents: Any, extra: Any) -> list[dict[str, Any]]:
    """``_subagents_with_middleware`` output as plain dicts, for key assertions."""
    return [dict(spec) for spec in _subagents_with_middleware(subagents, extra)]


class TestToolConfigInjection:
    def test_gemini_without_tool_choice_gets_validated_mode(self) -> None:
        """A subagent turn carries no tool_choice, which is what leaves Vertex on AUTO."""
        prepared = _PayloadHandlerMiddleware()._prepare_request(_request(_gemini()))

        assert prepared.model_settings["tool_config"] == VALIDATED_TOOL_CONFIG

    def test_gemini_with_tool_choice_is_left_alone(self) -> None:
        prepared = _PayloadHandlerMiddleware()._prepare_request(
            _request(_gemini(), tool_choice="any")
        )

        assert "tool_config" not in prepared.model_settings

    def test_response_format_is_left_alone(self) -> None:
        """A response format makes create_agent derive tool_choice="any" at bind
        time, after this middleware has run, so injecting a mode here would
        collide with a choice we never saw on the request."""
        prepared = _PayloadHandlerMiddleware()._prepare_request(
            _request(_gemini(), response_format=ToolStrategy({"type": "object"}))
        )

        assert "tool_config" not in prepared.model_settings

    def test_a_response_format_request_binds_the_way_create_agent_binds_it(
        self,
    ) -> None:
        """The main agent's call, reproduced: create_agent forces "any" for a
        ToolStrategy regardless of request.tool_choice."""
        model = _gemini()
        prepared = _PayloadHandlerMiddleware()._prepare_request(
            _request(model, response_format=ToolStrategy({"type": "object"}))
        )

        model.bind_tools([echo], tool_choice="any", **prepared.model_settings)

    def test_injected_config_binds_without_conflicting(self) -> None:
        """langchain_google_genai raises when tool_choice and tool_config collide."""
        model = _gemini()
        prepared = _PayloadHandlerMiddleware()._prepare_request(_request(model))

        model.bind_tools([echo], tool_choice=None, **prepared.model_settings)

    def test_non_gemini_model_is_untouched(self) -> None:
        prepared = _PayloadHandlerMiddleware()._prepare_request(
            _request(GenericFakeChatModel(messages=iter([])))
        )

        assert prepared.model_settings == {}

    def test_existing_model_settings_are_preserved(self) -> None:
        request = ModelRequest(
            model=_gemini(),
            messages=[],
            tools=[echo],
            model_settings={"temperature": 0},
        )

        prepared = _PayloadHandlerMiddleware()._prepare_request(request)

        assert prepared.model_settings["temperature"] == 0
        assert prepared.model_settings["tool_config"] == VALIDATED_TOOL_CONFIG


class TestStopReasonCheck:
    def test_malformed_function_call_raises(self) -> None:
        """Gemini reports the malformation here; without this it reads as a final answer."""
        middleware = _PayloadHandlerMiddleware()
        response = _response(
            AIMessage(
                content="",
                response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
            )
        )

        with pytest.raises(ChatModelError) as exc_info:
            middleware._validate_response(_request(_gemini()), response)

        assert "invalid function call" in exc_info.value.error_info.title.lower()

    def test_clean_finish_reason_passes(self) -> None:
        middleware = _PayloadHandlerMiddleware()
        response = _response(
            AIMessage(content="done", response_metadata={"finish_reason": "STOP"})
        )

        middleware._validate_response(_request(_gemini()), response)

    def test_non_gemini_finish_reason_is_not_checked_as_gemini(self) -> None:
        middleware = _PayloadHandlerMiddleware()
        response = _response(
            AIMessage(
                content="done",
                response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
            )
        )

        middleware._validate_response(
            _request(GenericFakeChatModel(messages=iter([]))), response
        )


class TestEmptyAnswerRejection:
    def test_empty_message_without_tool_calls_raises(self) -> None:
        middleware = _PayloadHandlerMiddleware()

        with pytest.raises(AgentRuntimeError) as exc_info:
            middleware._validate_response(
                _request(GenericFakeChatModel(messages=iter([]))),
                _response(AIMessage(content="")),
            )

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.LLM_INVALID_RESPONSE
        )
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM

    def test_whitespace_only_message_raises(self) -> None:
        middleware = _PayloadHandlerMiddleware()

        with pytest.raises(AgentRuntimeError):
            middleware._validate_response(
                _request(GenericFakeChatModel(messages=iter([]))),
                _response(AIMessage(content="   \n")),
            )

    def test_empty_message_with_tool_calls_passes(self) -> None:
        middleware = _PayloadHandlerMiddleware()
        response = _response(
            AIMessage(
                content="",
                tool_calls=[{"name": "echo", "args": {"text": "x"}, "id": "1"}],
            )
        )

        middleware._validate_response(
            _request(GenericFakeChatModel(messages=iter([]))), response
        )

    def test_reasoning_only_message_passes(self) -> None:
        """A thinking turn has no text and no tool calls, but is not a dead end."""
        middleware = _PayloadHandlerMiddleware()
        response = _response(
            AIMessage(content=[{"type": "reasoning", "reasoning": "working on it"}])
        )

        middleware._validate_response(
            _request(GenericFakeChatModel(messages=iter([]))), response
        )

    def test_structured_response_passes(self) -> None:
        """A structured answer arrives with the text already consumed by the tool call."""
        middleware = _PayloadHandlerMiddleware()

        middleware._validate_response(
            _request(GenericFakeChatModel(messages=iter([]))),
            _response(AIMessage(content=""), structured={"result": "ok"}),
        )


class TestWrapModelCall:
    def test_sync_shapes_request_and_checks_response(self) -> None:
        middleware = _PayloadHandlerMiddleware()
        seen: list[ModelRequest[Any]] = []

        def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
            seen.append(request)
            return _response(AIMessage(content="hi"))

        middleware.wrap_model_call(_request(_gemini()), handler)

        assert seen[0].model_settings["tool_config"] == VALIDATED_TOOL_CONFIG

    async def test_async_shapes_request_and_checks_response(self) -> None:
        middleware = _PayloadHandlerMiddleware()
        seen: list[ModelRequest[Any]] = []

        async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
            seen.append(request)
            return _response(AIMessage(content="hi"))

        await middleware.awrap_model_call(_request(_gemini()), handler)

        assert seen[0].model_settings["tool_config"] == VALIDATED_TOOL_CONFIG

    async def test_async_raises_on_malformed_call(self) -> None:
        middleware = _PayloadHandlerMiddleware()

        async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
            return _response(
                AIMessage(
                    content="",
                    response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
                )
            )

        with pytest.raises(ChatModelError):
            await middleware.awrap_model_call(_request(_gemini()), handler)


class TestSubagentWiring:
    def test_general_purpose_spec_is_added_with_the_middleware(self) -> None:
        """deepagents builds this subagent itself, so its spec is the only seam."""
        middleware = _PayloadHandlerMiddleware()

        specs = _specs([], [middleware])

        assert [spec["name"] for spec in specs] == ["general-purpose"]
        assert specs[0]["middleware"] == [middleware]

    def test_general_purpose_spec_inherits_parent_tools(self) -> None:
        """Omitting 'tools' is what makes deepagents pass the parent's tools down."""
        specs = _specs([], [_PayloadHandlerMiddleware()])

        assert "tools" not in specs[0]

    def test_caller_subagents_keep_their_own_middleware(self) -> None:
        existing = _PayloadHandlerMiddleware()
        ours = _PayloadHandlerMiddleware()
        spec: Any = {
            "name": "researcher",
            "description": "d",
            "system_prompt": "p",
            "middleware": [existing],
        }

        specs = _specs([spec], [ours])

        researcher = next(s for s in specs if s["name"] == "researcher")
        assert researcher["middleware"] == [existing, ours]

    def test_caller_general_purpose_override_is_not_duplicated(self) -> None:
        spec: Any = {
            "name": "general-purpose",
            "description": "custom",
            "system_prompt": "p",
        }

        specs = _specs([spec], [_PayloadHandlerMiddleware()])

        assert len(specs) == 1
        assert specs[0]["description"] == "custom"

    def test_compiled_subagent_is_passed_through(self) -> None:
        spec: Any = {"name": "compiled", "description": "d", "runnable": MagicMock()}

        specs = _specs([spec], [_PayloadHandlerMiddleware()])

        compiled = next(s for s in specs if s["name"] == "compiled")
        assert "middleware" not in compiled

    def test_builder_gives_the_middleware_to_agent_and_subagent(self) -> None:
        with patch(
            "uipath_langchain.agent.advanced.agent._create_deep_agent",
            return_value=MagicMock(),
        ) as mock_create:
            create_advanced_agent(model=GenericFakeChatModel(messages=iter([])))

        kwargs = mock_create.call_args.kwargs
        main = [
            m for m in kwargs["middleware"] if isinstance(m, _PayloadHandlerMiddleware)
        ]
        subagent = kwargs["subagents"][0]["middleware"]

        assert len(main) == 1
        assert main[0] is subagent[-1]


def test_bound_tools_are_filtered_to_basetools() -> None:
    """request.tools may hold provider built-in dicts alongside BaseTools."""
    request = ModelRequest(
        model=_gemini(),
        messages=[],
        tools=[echo, {"google_search": {}}],
    )

    prepared = _PayloadHandlerMiddleware()._prepare_request(request)

    assert prepared.model_settings["tool_config"] == VALIDATED_TOOL_CONFIG
    assert isinstance(request.tools[0], BaseTool)


def _general_purpose_spec(build: Callable[[], Any]) -> dict[str, Any]:
    """The general-purpose spec deepagents actually receives from ``build``.

    Nothing else here builds a real deep agent, so nothing else notices when a
    supplied spec stops matching the one deepagents would have assembled.
    """
    captured: dict[str, Any] = {}
    original = SubAgentMiddleware.__init__

    def record(self: Any, *args: Any, **kwargs: Any) -> None:
        captured["subagents"] = kwargs.get("subagents") or (args[0] if args else [])
        original(self, *args, **kwargs)

    with patch.object(SubAgentMiddleware, "__init__", record):
        build()
    return next(
        spec for spec in captured["subagents"] if spec["name"] == "general-purpose"
    )


class TestGeneralPurposeSubagentParity:
    """Supplying the spec ourselves opts out of the one deepagents assembles.

    Its two code paths are not identical, so anything the auto-added spec would
    have carried has to be restated on ours. These compare the two directly.
    """

    def _build(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        model = GenericFakeChatModel(messages=iter([]))
        baseline = _general_purpose_spec(
            lambda: create_deep_agent(
                model=model, system_prompt="p", tools=[echo], subagents=[], **kwargs
            )
        )
        ours = _general_purpose_spec(
            lambda: create_advanced_agent(
                model=model, system_prompt="p", tools=[echo], subagents=[], **kwargs
            )
        )
        return baseline, ours

    def test_middleware_matches_deepagents_plus_ours(self) -> None:
        baseline, ours = self._build()

        names = [m.name for m in ours["middleware"]]
        assert [n for n in names if n != _PayloadHandlerMiddleware.__name__] == [
            m.name for m in baseline["middleware"]
        ]
        assert _PayloadHandlerMiddleware.__name__ in names

    def test_skills_reach_the_subagent(self) -> None:
        """Restated on the spec: deepagents reads a supplied spec's skills from it."""
        baseline, ours = self._build(skills=["/skills"])

        assert "SkillsMiddleware" in [m.name for m in baseline["middleware"]]
        assert "SkillsMiddleware" in [m.name for m in ours["middleware"]]

    def test_prompt_and_tools_match(self) -> None:
        baseline, ours = self._build()

        assert ours["system_prompt"] == baseline["system_prompt"]
        assert [t.name for t in ours["tools"]] == [t.name for t in baseline["tools"]]
