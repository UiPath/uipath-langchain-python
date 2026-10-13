"""Tests for LLM node tool call filtering functionality."""

import copy
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock, patch

import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.content import create_text_block, create_tool_call
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.core.feature_flags import FeatureFlags
from uipath.llm_client import UiPathAPIError, UiPathError, UiPathLLMErrorCode
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.llm_node import (
    BEDROCK_PROMPT_CACHING_FF,
    AutoToolChoiceLLMNode,
    ConversationalLLMNode,
    ForcedToolChoiceLLMNode,
    create_llm_node,
)
from uipath_langchain.agent.react.tools import create_flow_control_tools
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.chat.exceptions import ChatModelError


class _StubAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI subclass that bypasses Pydantic validation for testing.

    Keeps AzureChatOpenAI in the MRO for handler resolution while allowing
    arbitrary attribute assignment (e.g. setting Mock methods).
    """

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


class TestLLMNodeParallelToolCalls:
    """Test that parallel_tool_calls parameter flows to model.bind_tools()."""

    mock_model: Any

    def setup_method(self):
        self.mock_model = _StubAzureChatOpenAI.model_construct()
        self.mock_model.bind_tools = Mock(return_value=self.mock_model)
        self.mock_model.bind = Mock(return_value=self.mock_model)

        self.regular_tool = Mock(spec=BaseTool)
        self.regular_tool.name = "regular_tool"

        self.test_state = AgentGraphState(messages=[HumanMessage(content="Test")])

    @pytest.mark.asyncio
    async def test_parallel_true_passes_kwarg(self):
        mock_response = AIMessage(content="done", tool_calls=[])
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(
            self.mock_model, [self.regular_tool], parallel_tool_calls=True
        )
        await llm_node(self.test_state)

        self.mock_model.bind_tools.assert_called_once()
        _, kwargs = self.mock_model.bind_tools.call_args
        assert kwargs["parallel_tool_calls"] is True

    @pytest.mark.asyncio
    async def test_parallel_false_passes_kwarg(self):
        mock_response = AIMessage(content="done", tool_calls=[])
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(
            self.mock_model,
            [self.regular_tool],
            parallel_tool_calls=False,
        )
        await llm_node(self.test_state)

        self.mock_model.bind_tools.assert_called_once()
        _, kwargs = self.mock_model.bind_tools.call_args
        assert kwargs["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_default_is_true(self):
        mock_response = AIMessage(content="done", tool_calls=[])
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(self.mock_model, [self.regular_tool])
        await llm_node(self.test_state)

        self.mock_model.bind_tools.assert_called_once()
        _, kwargs = self.mock_model.bind_tools.call_args
        assert kwargs["parallel_tool_calls"] is True

    @pytest.mark.asyncio
    async def test_non_openai_provider_no_parallel_kwarg(self):
        """Default handler does not include parallel_tool_calls in binding kwargs."""
        mock_model = Mock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = mock_model
        mock_model.bind.return_value = mock_model

        mock_response = AIMessage(content="done", tool_calls=[])
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(
            mock_model,
            [self.regular_tool],
            parallel_tool_calls=False,
        )
        await llm_node(self.test_state)

        mock_model.bind_tools.assert_called_once()
        _, kwargs = mock_model.bind_tools.call_args
        assert "parallel_tool_calls" not in kwargs


class TestLLMNodeToolCallFiltering:
    """Test cases for LLM node tool call filtering integration."""

    mock_model: Any

    def setup_method(self):
        """Set up test fixtures."""
        self.regular_tool = Mock(spec=BaseTool)
        self.regular_tool.name = "regular_tool"

        self.mock_model = _StubAzureChatOpenAI.model_construct()
        self.mock_model.bind_tools = Mock(return_value=self.mock_model)
        self.mock_model.bind = Mock(return_value=self.mock_model)

        self.test_state = AgentGraphState(messages=[HumanMessage(content="Test query")])

    @pytest.mark.asyncio
    async def test_single_flow_control_call_not_filtered(self):
        """Single flow control calls should not be filtered by LLM node."""
        # Mock response with single flow control call
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("I need to end execution"),
                create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="call_1"),
            ],
            tool_calls=[
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {},
                    "id": "call_1",
                }
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify single flow control call is not filtered
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == END_EXECUTION_TOOL.name

    @pytest.mark.asyncio
    async def test_parallel_flow_control_calls_filtered(self):
        """Flow control calls in parallel should be filtered by LLM node."""
        # Mock response with parallel calls including flow control
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("Using multiple tools"),
                create_tool_call(name="regular_tool", args={}, id="call_1"),
                create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="call_2"),
            ],
            tool_calls=[
                {"name": "regular_tool", "args": {}, "id": "call_1"},
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {},
                    "id": "call_2",
                },
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify flow control call was filtered out
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == "regular_tool"

        # Verify content blocks were also updated
        tool_call_blocks = [
            block
            for block in response_message.content_blocks
            if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "regular_tool"

    @pytest.mark.asyncio
    async def test_no_flow_control_calls_unchanged(self):
        """Regular tool calls without flow control should remain unchanged."""
        # Mock response with only regular calls
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("Using regular tools"),
                create_tool_call(name="regular_tool", args={}, id="call_1"),
                create_tool_call(name="another_tool", args={}, id="call_2"),
            ],
            tool_calls=[
                {"name": "regular_tool", "args": {}, "id": "call_1"},
                {"name": "another_tool", "args": {}, "id": "call_2"},
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify no filtering occurred
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 2
        assert response_message.tool_calls[0]["name"] == "regular_tool"
        assert response_message.tool_calls[1]["name"] == "another_tool"

    @pytest.mark.asyncio
    async def test_multiple_flow_control_calls_all_filtered(self):
        """Multiple flow control calls in parallel should all be filtered."""
        # Mock response with regular and multiple flow control calls
        mock_response = AIMessage(
            content_blocks=[
                create_text_block("Complex scenario"),
                create_tool_call(name="regular_tool", args={}, id="call_1"),
                create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="call_2"),
                create_tool_call(name=RAISE_ERROR_TOOL.name, args={}, id="call_3"),
            ],
            tool_calls=[
                {"name": "regular_tool", "args": {}, "id": "call_1"},
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {},
                    "id": "call_2",
                },
                {"name": RAISE_ERROR_TOOL.name, "args": {}, "id": "call_3"},
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        # Create LLM node
        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        # Execute node
        result = await llm_node(self.test_state)

        # Verify only regular tool call remains
        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == "regular_tool"

        # Verify content blocks were updated accordingly
        tool_call_blocks = [
            block
            for block in response_message.content_blocks
            if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "regular_tool"

    @pytest.mark.asyncio
    async def test_end_execution_and_raise_error_keeps_only_raise_error(self):
        """When only end_execution and raise_error are called, keep only raise_error."""
        mock_response = AIMessage(
            content_blocks=[
                create_tool_call(
                    name=END_EXECUTION_TOOL.name,
                    args={"result": "done"},
                    id="call_1",
                ),
                create_tool_call(
                    name=RAISE_ERROR_TOOL.name,
                    args={"message": "conflict"},
                    id="call_2",
                ),
            ],
            tool_calls=[
                {
                    "name": END_EXECUTION_TOOL.name,
                    "args": {"result": "done"},
                    "id": "call_1",
                },
                {
                    "name": RAISE_ERROR_TOOL.name,
                    "args": {"message": "conflict"},
                    "id": "call_2",
                },
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        result = await llm_node(self.test_state)

        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == RAISE_ERROR_TOOL.name

    @pytest.mark.asyncio
    async def test_multiple_raise_error_calls_keeps_only_first(self):
        """When raise_error is called multiple times, keep only the first."""
        mock_response = AIMessage(
            content_blocks=[
                create_tool_call(
                    name=RAISE_ERROR_TOOL.name,
                    args={"message": "first error"},
                    id="call_1",
                ),
                create_tool_call(
                    name=RAISE_ERROR_TOOL.name,
                    args={"message": "second error"},
                    id="call_2",
                ),
            ],
            tool_calls=[
                {
                    "name": RAISE_ERROR_TOOL.name,
                    "args": {"message": "first error"},
                    "id": "call_1",
                },
                {
                    "name": RAISE_ERROR_TOOL.name,
                    "args": {"message": "second error"},
                    "id": "call_2",
                },
            ],
        )
        self.mock_model.ainvoke = AsyncMock(return_value=mock_response)

        llm_node = create_llm_node(self.mock_model, [self.regular_tool])

        result = await llm_node(self.test_state)

        response_message = result["messages"][0]
        assert len(response_message.tool_calls) == 1
        assert response_message.tool_calls[0]["name"] == RAISE_ERROR_TOOL.name
        assert response_message.tool_calls[0]["args"]["message"] == "first error"


class TestLLMNodeProviderErrorHandling:
    """llm_node maps provider HTTP errors to AgentRuntimeError (new + legacy clients)."""

    mock_model: Any

    def setup_method(self):
        self.mock_model = _StubAzureChatOpenAI.model_construct()
        self.mock_model.bind_tools = Mock(return_value=self.mock_model)
        self.mock_model.bind = Mock(return_value=self.mock_model)
        self.tool = Mock(spec=BaseTool)
        self.tool.name = "regular_tool"
        self.state = AgentGraphState(messages=[HumanMessage(content="Test")])

    def _node_raising(self, exc: BaseException):
        self.mock_model.ainvoke = AsyncMock(side_effect=exc)
        return create_llm_node(self.mock_model, [self.tool])

    @staticmethod
    def _http_403() -> httpx.Response:
        """A genuine gateway licensing 403, carrying its errorCode marker."""
        request = httpx.Request("POST", "http://gateway/")
        return httpx.Response(
            403,
            request=request,
            json={
                "title": "License not available",
                "status": 403,
                "detail": "need AGU",
                "errorCode": 10000,
            },
        )

    @staticmethod
    def _http_403_unmarked() -> httpx.Response:
        """A third-party 403 relayed by the gateway: no licensing marker.

        The shape both PC-5000 and SRE-654983 actually hit.
        """
        request = httpx.Request("POST", "http://gateway/")
        return httpx.Response(
            403,
            request=request,
            text="<!doctype html><title>403</title>403 Forbidden",
            headers={"content-type": "text/html"},
        )

    @pytest.mark.asyncio
    async def test_new_client_uipath_api_error_maps_to_license(self):
        # New LLM clients raise a normalized UiPathAPIError -> except UiPathAPIError.
        node = self._node_raising(UiPathAPIError.from_response(self._http_403()))

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(self.state)

        info = exc_info.value.error_info
        assert info.status == 403
        assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)
        assert info.detail == "need AGU"

    @pytest.mark.asyncio
    async def test_legacy_raw_provider_error_is_normalized_and_mapped(self):
        # Legacy clients (use_new_llm_clients=False) raise raw provider exceptions
        # -> except Exception -> as_uipath_error normalizes -> mapped.
        raw = openai.PermissionDeniedError(
            "Forbidden",
            response=self._http_403(),
            body={"status": 403, "detail": "need AGU"},
        )
        node = self._node_raising(raw)

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(self.state)

        info = exc_info.value.error_info
        assert info.status == 403
        assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)

    @staticmethod
    def _http_400() -> httpx.Response:
        """A gateway 400 as seen in prod: max_tokens above the model's limit."""
        request = httpx.Request("POST", "http://gateway/")
        return httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "message": (
                        "max_tokens is too large: 65535. This model supports at "
                        "most 32768 completion tokens."
                    ),
                    "code": "invalid_value",
                    "param": "max_tokens",
                }
            },
        )

    @pytest.mark.asyncio
    async def test_new_client_400_maps_to_user_without_the_provider_body(self):
        # Previously this reached telemetry as the two words "Bad Request",
        # categorized Unknown. It is now User, with a canned actionable detail
        # and no provider text.
        node = self._node_raising(UiPathAPIError.from_response(self._http_400()))

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(self.state)

        info = exc_info.value.error_info
        assert info.status == 400
        assert info.category == UiPathErrorCategory.USER
        assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST.value)
        assert info.detail != "Bad Request"
        assert "65535" not in info.detail

    @pytest.mark.asyncio
    async def test_legacy_400_maps_to_user(self):
        raw = openai.BadRequestError(
            "Bad Request",
            response=self._http_400(),
            body={"error": {"message": "max_tokens is too large: 65535."}},
        )
        node = self._node_raising(raw)

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(self.state)

        info = exc_info.value.error_info
        assert info.status == 400
        assert info.category == UiPathErrorCategory.USER
        assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST.value)

    @pytest.mark.asyncio
    async def test_unmarked_403_maps_to_provider_forbidden_not_license(self):
        # Regression for PC-5000 / SRE-654983: an edge or BYOM endpoint refuses
        # the request and the gateway relays it verbatim. Nothing licensing
        # happened, so nothing licensing may be reported.
        node = self._node_raising(
            UiPathAPIError.from_response(self._http_403_unmarked())
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(self.state)

        info = exc_info.value.error_info
        assert info.status == 403
        assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_FORBIDDEN.value)
        # The relayed body is not quoted back: detail is the HTTP reason phrase.
        assert info.detail == "Forbidden"

    @pytest.mark.asyncio
    async def test_new_client_unsupported_mime_error_maps_to_file_error(self):
        node = self._node_raising(
            UiPathError(
                error_code=UiPathLLMErrorCode.UNSUPPORTED_MIME_TYPE,
                detail="Unsupported MIME type: application/x-foo",
            )
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(self.state)

        info = exc_info.value.error_info
        assert info.category == UiPathErrorCategory.USER
        assert info.code.endswith(AgentRuntimeErrorCode.FILE_ERROR.value)
        assert "application/x-foo" in info.detail

    @pytest.mark.asyncio
    async def test_unmapped_uipath_error_propagates_unchanged(self):
        err = UiPathError(error_code="SOME_OTHER_CODE", detail="unrelated")
        node = self._node_raising(err)

        with pytest.raises(UiPathError) as exc_info:
            await node(self.state)

        assert exc_info.value is err

    @pytest.mark.asyncio
    async def test_non_http_error_propagates_unchanged(self):
        # No HTTP status -> as_uipath_error yields a non-UiPathAPIError -> re-raised.
        node = self._node_raising(ValueError("boom"))

        with pytest.raises(ValueError, match="boom"):
            await node(self.state)


class TestForcedExtractionEscalation:
    """Stall accounting in the LLM node. Tool usage is forced every turn. Anthropic
    thinking models can't be plain-forced (turn 1's forced `any` is downgraded to `auto`
    by the handler so they reason), so their first stall retries via a thinking-off
    extraction call; every other provider forces from turn 1 and is never extracted
    (forcing works for it, and the extraction's reasoning-strip would break it, e.g.
    OpenAI 400s on a continuation with its reasoning removed). A second stall fails
    deterministically. Not gated on tool_choice='auto', since forcing can be downgraded
    on the wire (Bedrock handlers, langchain_anthropic)."""

    def _thinking_model(self) -> Any:
        model: Any = _StubAzureChatOpenAI.model_construct()
        model.thinking = {
            "type": "adaptive"
        }  # thinking active; OpenAI MRO won't downgrade
        model.bind_tools = Mock(return_value=model)
        model.bind = Mock(return_value=model)
        return model

    def _plain_model(self) -> Any:
        model: Any = _StubAzureChatOpenAI.model_construct()
        model.bind_tools = Mock(return_value=model)
        model.bind = Mock(return_value=model)
        return model

    def _stalled_state(self, stalls: int = 1) -> AgentGraphState:
        prior = AIMessage(
            content=[
                {"type": "reasoning_content", "reasoning_content": {"text": "think"}},
                {"type": "text", "text": "answer"},
            ]
        )
        return AgentGraphState(
            messages=[HumanMessage(content="q"), *([prior] * stalls)]
        )

    async def _run_capture(
        self, model: Any, tool_choice: Literal["auto", "any"] = "auto"
    ) -> list[Any]:
        captured: dict[str, Any] = {}

        async def fake_ainvoke(msgs: Any) -> AIMessage:
            captured["msgs"] = msgs
            return AIMessage(
                content="",
                tool_calls=[
                    create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="c1")
                ],
            )

        model.ainvoke = AsyncMock(side_effect=fake_ainvoke)
        tool = Mock(spec=BaseTool)
        tool.name = "t"
        node = create_llm_node(model, [tool], tool_choice=tool_choice)
        await node(self._stalled_state())
        return captured["msgs"]

    @pytest.mark.asyncio
    async def test_thinking_model_stall_strips_reasoning_and_nudges(self) -> None:
        """Escalation fires: reasoning stripped from the stalled turn, and the request
        ends on a user turn (not an assistant prefill) so native accepts the forced call."""
        msgs = await self._run_capture(self._thinking_model())
        # stalled assistant turn: reasoning block gone, text kept
        assert msgs[-2].content == [{"type": "text", "text": "answer"}]
        # request ends on a user turn so the forced tool call is accepted
        assert isinstance(msgs[-1], HumanMessage)

    @pytest.mark.asyncio
    async def test_plain_model_stall_is_forced_not_extracted(self) -> None:
        """A non-Anthropic model that stalls is plain-forced, never extracted: forcing
        works for it directly, and the extraction's reasoning-strip would break it."""
        model = self._plain_model()
        model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="",
                tool_calls=[
                    create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="c1")
                ],
            )
        )
        tool = Mock(spec=BaseTool)
        tool.name = "t"
        with patch(
            "uipath_langchain.agent.react.llm_node.build_extraction_call"
        ) as spy:
            await create_llm_node(model, [tool])(self._stalled_state())
        spy.assert_not_called()
        assert model.bind_tools.call_args.kwargs["tool_choice"] == "any"

    @pytest.mark.asyncio
    async def test_escalates_when_tool_choice_configured_any(self) -> None:
        """AgentGraphConfig(tool_choice='any') keeps the termination guarantee: the
        extraction retry must not be gated on tool_choice starting as 'auto'."""
        msgs = await self._run_capture(self._thinking_model(), tool_choice="any")
        assert isinstance(msgs[-1], HumanMessage)

    @pytest.mark.asyncio
    async def test_no_stall_does_not_escalate(self) -> None:
        """First turn (no prior tool-less turn) must not force extraction."""
        model = self._thinking_model()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="reasoning..."))
        state = AgentGraphState(messages=[HumanMessage(content="q")])
        tool = Mock(spec=BaseTool)
        tool.name = "t"
        with patch(
            "uipath_langchain.agent.react.llm_node.build_extraction_call"
        ) as spy:
            await create_llm_node(model, [tool])(state)
        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_stall_after_extraction_raises_thinking_limit(self) -> None:
        """A model that stalls again after the forced-extraction retry fails fast
        with a SYSTEM diagnostic instead of looping to the max-iterations limit."""
        model = self._thinking_model()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="still stalling"))
        tool = Mock(spec=BaseTool)
        tool.name = "t"
        node = create_llm_node(model, [tool])

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(self._stalled_state(stalls=2))

        info = exc_info.value.error_info
        assert info.code.endswith(AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED.value)
        assert info.category == UiPathErrorCategory.SYSTEM
        model.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_plain_model_forces_from_first_turn(self) -> None:
        """A non-Anthropic model forces tool_choice from turn 1 (it reasons and calls
        the tool in the same forced turn), so no reasoning buffer is needed."""
        model = self._plain_model()
        model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="",
                tool_calls=[
                    create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="c1")
                ],
            )
        )
        tool = Mock(spec=BaseTool)
        tool.name = "t"
        node = create_llm_node(model, [tool])
        await node(AgentGraphState(messages=[HumanMessage(content="q")]))
        assert model.bind_tools.call_args.kwargs["tool_choice"] == "any"


class _StubChatAnthropic(ChatAnthropic):
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


class _Output(BaseModel):
    answer: str


class _UntypedListOutput(BaseModel):
    sources: list[Any]


_REJECTS_FORCING = {"shouldSkipForcedToolChoice": True}


class _StubChatBedrockConverse(ChatBedrockConverse):
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


class TestBedrockConverseFinishWithoutForcing:
    """Bedrock Converse Claude finishes the same way as Vertex."""

    @pytest.mark.asyncio
    async def test_text_answer_finishes_by_calling_end_execution(self) -> None:
        model: Any = _StubChatBedrockConverse.model_construct(
            model_id="anthropic.claude-opus-5-5-v1:0",
        )
        model.model_details = _REJECTS_FORCING
        model.bind_tools = Mock(return_value=model)
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content=[{"type": "text", "text": "Lisbon is warmer."}]),
                AIMessage(
                    content=[],
                    tool_calls=[
                        create_tool_call(
                            name=END_EXECUTION_TOOL.name,
                            args={"answer": "Lisbon"},
                            id="t1",
                        )
                    ],
                ),
            ]
        )
        search = StructuredTool(
            name="search",
            description="search",
            args_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            func=lambda **_: None,
        )
        tools = [search, *create_flow_control_tools(_Output)]

        result = await create_llm_node(model, tools)(
            AgentGraphState(messages=[HumanMessage("q")])
        )

        finish_binding = model.bind_tools.call_args_list[1]
        assert [t.name for t in finish_binding.args[0]] == [
            END_EXECUTION_TOOL.name,
            RAISE_ERROR_TOOL.name,
        ]
        assert finish_binding.kwargs["tool_choice"] == "auto"
        assert "output_config" not in finish_binding.kwargs
        assert result["messages"][0].tool_calls[0]["args"] == {"answer": "Lisbon"}


class TestFinishWithoutForcing:
    """Opus 5.5 rejects forced tool choice, so its turns run with tool_choice auto and a
    text answer is turned into end_execution by one finish call."""

    def _model(self, *responses: BaseMessage) -> Any:
        model: Any = _StubChatAnthropic.model_construct(model="claude-opus-5-5")
        model.model_details = _REJECTS_FORCING
        model.bind_tools = Mock(return_value=model)
        model.ainvoke = AsyncMock(side_effect=list(responses))
        return model

    def _tools(self) -> list[BaseTool]:
        search = StructuredTool(
            name="search",
            description="search",
            args_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            func=lambda **_: None,
        )
        return [search, *create_flow_control_tools(_Output)]

    def _node(self, model: Any) -> Any:
        return create_llm_node(model, self._tools())

    @staticmethod
    def _prose(stop_reason: str = "end_turn") -> AIMessage:
        return AIMessage(
            content=[{"type": "text", "text": "Lisbon is warmer (27 °C)."}],
            response_metadata={"stop_reason": stop_reason},
        )

    @pytest.mark.asyncio
    async def test_turns_run_with_auto(self) -> None:
        model = self._model(
            AIMessage(
                content="",
                tool_calls=[create_tool_call(name="search", args={}, id="c1")],
            )
        )

        await self._node(model)(AgentGraphState(messages=[HumanMessage("q")]))

        assert model.bind_tools.call_args.kwargs["tool_choice"] == "auto"
        assert "output_config" not in model.bind_tools.call_args.kwargs

    @pytest.mark.asyncio
    async def test_anthropic_text_answer_finishes_by_calling_end_execution(
        self,
    ) -> None:
        reply = AIMessage(
            content=[],
            tool_calls=[
                create_tool_call(
                    name=END_EXECUTION_TOOL.name, args={"answer": "Lisbon"}, id="t1"
                )
            ],
            response_metadata={"stop_reason": "tool_use"},
        )
        model = self._model(self._prose(), reply)

        result = await self._node(model)(AgentGraphState(messages=[HumanMessage("q")]))

        assert model.ainvoke.await_count == 2
        finish_binding = model.bind_tools.call_args_list[1]
        assert [t.name for t in finish_binding.args[0]] == [
            END_EXECUTION_TOOL.name,
            RAISE_ERROR_TOOL.name,
        ]
        assert finish_binding.kwargs["tool_choice"] == "auto"
        assert "output_config" not in finish_binding.kwargs
        sent = model.ainvoke.call_args_list[1].args[0]
        assert sent[-2].text == "Lisbon is warmer (27 °C)."
        assert isinstance(sent[-1], HumanMessage)
        assert END_EXECUTION_TOOL.name in sent[-1].text
        [response] = result["messages"]
        assert [(tc["name"], tc["args"]) for tc in response.tool_calls] == [
            (END_EXECUTION_TOOL.name, {"answer": "Lisbon"})
        ]

    def _untyped_list_node(self, model: Any) -> Any:
        tools = [self._tools()[0], *create_flow_control_tools(_UntypedListOutput)]
        return create_llm_node(model, tools)

    @pytest.mark.asyncio
    async def test_untyped_output_finishes_by_calling_end_execution(
        self,
    ) -> None:
        reply = AIMessage(
            content=[{"type": "thinking", "thinking": "", "signature": "s"}],
            tool_calls=[
                create_tool_call(
                    name=END_EXECUTION_TOOL.name,
                    args={"sources": ["https://example.com"]},
                    id="toolu_1",
                )
            ],
            response_metadata={"stop_reason": "tool_use"},
        )
        model = self._model(self._prose(), reply)

        result = await self._untyped_list_node(model)(
            AgentGraphState(messages=[HumanMessage("q")])
        )

        assert model.ainvoke.await_count == 2
        finish_binding = model.bind_tools.call_args_list[1]
        assert [t.name for t in finish_binding.args[0]] == [
            END_EXECUTION_TOOL.name,
            RAISE_ERROR_TOOL.name,
        ]
        assert finish_binding.kwargs["tool_choice"] == "auto"
        assert "output_config" not in finish_binding.kwargs
        sent = model.ainvoke.call_args_list[1].args[0]
        assert sent[-2].text == "Lisbon is warmer (27 °C)."
        assert isinstance(sent[-1], HumanMessage)
        assert END_EXECUTION_TOOL.name in sent[-1].text
        [response] = result["messages"]
        assert [(tc["name"], tc["args"]) for tc in response.tool_calls] == [
            (END_EXECUTION_TOOL.name, {"sources": ["https://example.com"]})
        ]
        assert response.content == [
            {"type": "thinking", "thinking": "", "signature": "s"}
        ]

    @pytest.mark.asyncio
    async def test_finish_call_without_end_execution_fails_the_run(self) -> None:
        model = self._model(self._prose(), self._prose())

        with pytest.raises(AgentRuntimeError) as exc_info:
            await self._untyped_list_node(model)(
                AgentGraphState(messages=[HumanMessage("q")])
            )

        assert model.ainvoke.await_count == 2
        assert exc_info.value.error_info.code.endswith(
            AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED.value
        )

    @pytest.mark.asyncio
    async def test_empty_finish_call_fails_the_run(self) -> None:
        reply = AIMessage(
            content=[],
            tool_calls=[
                create_tool_call(
                    name=END_EXECUTION_TOOL.name, args={"answer": None}, id="t1"
                )
            ],
        )
        model = self._model(self._prose(), reply)

        with pytest.raises(AgentRuntimeError) as exc_info:
            await self._node(model)(AgentGraphState(messages=[HumanMessage("q")]))

        assert exc_info.value.error_info.code.endswith(
            AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED.value
        )

    @pytest.mark.asyncio
    async def test_tool_calls_need_no_finish_call(self) -> None:
        model = self._model(
            AIMessage(
                content=[{"type": "text", "text": "Looking it up."}],
                tool_calls=[create_tool_call(name="search", args={"q": "x"}, id="c1")],
                response_metadata={"stop_reason": "tool_use"},
            )
        )

        result = await self._node(model)(AgentGraphState(messages=[HumanMessage("q")]))

        assert model.ainvoke.await_count == 1
        assert result["messages"][0].tool_calls[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_text_answer_finishes_whatever_the_stop_reason(self) -> None:
        reply = AIMessage(
            content=[],
            tool_calls=[
                create_tool_call(
                    name=END_EXECUTION_TOOL.name, args={"answer": "Lisbon"}, id="t1"
                )
            ],
        )
        model = self._model(self._prose("stop_sequence"), reply)

        result = await self._node(model)(AgentGraphState(messages=[HumanMessage("q")]))

        assert model.ainvoke.await_count == 2
        [response] = result["messages"]
        assert response.tool_calls[0]["args"] == {"answer": "Lisbon"}

    @pytest.mark.asyncio
    async def test_truncated_finish_call_raises(self) -> None:
        truncated = AIMessage(
            content=[{"type": "text", "text": "..."}],
            response_metadata={"stop_reason": "max_tokens"},
        )
        model = self._model(self._prose(), truncated)

        with pytest.raises(ChatModelError):
            await self._node(model)(AgentGraphState(messages=[HumanMessage("q")]))

    @pytest.mark.asyncio
    async def test_finish_call_keeps_parallel_tool_calls_and_strict_mode(
        self,
    ) -> None:
        model = self._model(
            self._prose(),
            AIMessage(
                content=[],
                tool_calls=[
                    create_tool_call(
                        name=END_EXECUTION_TOOL.name, args={"answer": "L"}, id="t1"
                    )
                ],
            ),
        )
        node = create_llm_node(
            model, self._tools(), parallel_tool_calls=False, strict_mode=True
        )

        await node(AgentGraphState(messages=[HumanMessage("q")]))

        finish_kwargs = model.bind_tools.call_args_list[1].kwargs
        assert finish_kwargs["parallel_tool_calls"] is False
        assert finish_kwargs["strict"] is True

    @pytest.mark.asyncio
    async def test_finish_reply_that_is_not_an_ai_message_is_invalid(self) -> None:
        model = self._model(self._prose(), HumanMessage("not an AI message"))

        with pytest.raises(AgentRuntimeError) as exc_info:
            await self._node(model)(AgentGraphState(messages=[HumanMessage("q")]))

        assert exc_info.value.error_info.code.endswith(
            AgentRuntimeErrorCode.LLM_INVALID_RESPONSE.value
        )

    @pytest.mark.asyncio
    async def test_a_tool_less_turn_already_in_the_history_fails_visibly(self) -> None:
        model = self._model()
        state = AgentGraphState(
            messages=[
                HumanMessage("q"),
                AIMessage(
                    content=[{"type": "thinking", "thinking": "", "signature": "s"}]
                ),
            ]
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            await self._node(model)(state)

        error = exc_info.value.error_info
        assert error.code.endswith(AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED.value)
        assert END_EXECUTION_TOOL.name in error.detail
        assert RAISE_ERROR_TOOL.name in error.detail
        assert "forced extraction" not in error.detail
        assert "asked to finish" not in error.detail
        model.ainvoke.assert_not_awaited()


class TestOtherClaudeThinkingModelsKeepTheForcedRetry:
    @pytest.mark.asyncio
    async def test_stall_is_retried_with_thinking_off_and_forced(self) -> None:
        model: Any = _StubChatAnthropic.model_construct(
            model="claude-opus-4-6", thinking={"type": "adaptive"}
        )
        model.bind_tools = Mock(return_value=model)
        model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="",
                tool_calls=[
                    create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="c1")
                ],
            )
        )
        tool = Mock(spec=BaseTool)
        tool.name = "t"
        stalled = AIMessage(content=[{"type": "text", "text": "answer"}])

        await create_llm_node(model, [tool])(
            AgentGraphState(messages=[HumanMessage(content="q"), stalled])
        )

        assert model.bind_tools.call_args.kwargs["tool_choice"] == "any"
        sent = model.ainvoke.call_args.args[0]
        assert isinstance(sent[-1], HumanMessage)
        assert len(sent) == 3


class TestFinishWithoutForcingCanRaiseError:
    @pytest.mark.asyncio
    async def test_failure_answered_in_text_is_raised(self) -> None:
        model: Any = _StubChatAnthropic.model_construct(model="claude-opus-5-5")
        model.model_details = _REJECTS_FORCING
        model.bind_tools = Mock(return_value=model)
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content=[{"type": "text", "text": "No such city."}]),
                AIMessage(
                    content=[],
                    tool_calls=[
                        create_tool_call(
                            name=RAISE_ERROR_TOOL.name,
                            args={"message": "No such city."},
                            id="t1",
                        )
                    ],
                ),
            ]
        )

        result = await create_llm_node(model, create_flow_control_tools(_Output))(
            AgentGraphState(messages=[HumanMessage("q")])
        )

        sent = model.ainvoke.call_args_list[1].args[0]
        assert RAISE_ERROR_TOOL.name in sent[-1].text
        [response] = result["messages"]
        assert [(tc["name"], tc["args"]) for tc in response.tool_calls] == [
            (RAISE_ERROR_TOOL.name, {"message": "No such city."})
        ]


class TestConversationalLLMNode:
    """Conversational agents bind the configured tool_choice and are never forced."""

    @staticmethod
    def _model() -> Any:
        model: Any = _StubChatAnthropic.model_construct(model="claude-opus-5-5")
        model.bind_tools = Mock(return_value=model)
        model.ainvoke = AsyncMock(
            return_value=AIMessage(content=[{"type": "text", "text": "Hello!"}])
        )
        return model

    @staticmethod
    def _tools() -> list[BaseTool]:
        return [
            StructuredTool(
                name="search",
                description="search",
                args_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                func=lambda **_: None,
            )
        ]

    @pytest.mark.asyncio
    async def test_binds_the_configured_tool_choice(self) -> None:
        for tool_choice in ("auto", "any"):
            model = self._model()
            node = create_llm_node(
                model, self._tools(), is_conversational=True, tool_choice=tool_choice
            )

            await node(AgentGraphState(messages=[HumanMessage("hi")]))

            assert model.bind_tools.call_args.kwargs["tool_choice"] == tool_choice


class TestCreateLlmNode:
    """The factory picks the LLM node by discovery's shouldSkipForcedToolChoice."""

    @staticmethod
    def _tools() -> list[BaseTool]:
        return create_flow_control_tools(_Output)

    @staticmethod
    def _model(model_details: dict[str, Any] | None) -> Any:
        model: Any = _StubChatAnthropic.model_construct(model="claude-opus-5-5")
        model.model_details = model_details
        return model

    def test_models_that_reject_forcing_get_the_auto_tool_choice_node(self) -> None:
        node = create_llm_node(self._model(_REJECTS_FORCING), self._tools())
        assert isinstance(node, AutoToolChoiceLLMNode)

    def test_models_without_the_discovery_flag_get_the_forced_node(self) -> None:
        for model_details in (None, {}, {"shouldSkipForcedToolChoice": False}):
            node = create_llm_node(self._model(model_details), self._tools())
            assert isinstance(node, ForcedToolChoiceLLMNode), model_details

    def test_conversational_agents_get_the_conversational_node(self) -> None:
        for model_details in (_REJECTS_FORCING, None):
            node = create_llm_node(
                self._model(model_details), self._tools(), is_conversational=True
            )
            assert isinstance(node, ConversationalLLMNode), model_details

    def test_agents_without_tools_get_the_conversational_node(self) -> None:
        model = _StubChatAnthropic.model_construct(model="claude-opus-4-6")
        assert isinstance(create_llm_node(model, []), ConversationalLLMNode)


def _converse_text(text: str) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
        "metrics": {"latencyMs": 1},
    }


def _converse_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "c1", "name": name, "input": args}}
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
        "metrics": {"latencyMs": 1},
    }


def _converse_stream_tool_call(name: str) -> dict[str, Any]:
    return {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "c1", "name": name}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": "{}"}},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                    "metrics": {"latencyMs": 1},
                }
            },
        ]
    }


def _cache_points(request: dict[str, Any]) -> dict[str, int]:
    def count(blocks: list[Any]) -> int:
        points = [b for b in blocks if isinstance(b, dict) and "cachePoint" in b]
        assert points == [{"cachePoint": {"type": "default"}}] * len(points)
        return len(points)

    *earlier, last = request["messages"]
    return {
        "tools": count(request.get("toolConfig", {}).get("tools", [])),
        "system": count(request.get("system", [])),
        "earlier_messages": sum(count(m["content"]) for m in earlier),
        "last_message": count(last["content"]),
    }


_ALL_MARKED = {"tools": 1, "system": 1, "earlier_messages": 0, "last_message": 1}
_NONE_MARKED = {"tools": 0, "system": 0, "earlier_messages": 0, "last_message": 0}


class TestBedrockPromptCaching:
    """With EnableBedrockPromptCaching on, Bedrock Converse turns carry cachePoint
    markers so the repeated prompt prefix of a run is read from Bedrock's cache."""

    @pytest.fixture(autouse=True)
    def _reset_flags(self) -> Any:
        FeatureFlags.reset_flags()
        yield
        FeatureFlags.reset_flags()

    @staticmethod
    def _enable() -> None:
        FeatureFlags.configure_flags({BEDROCK_PROMPT_CACHING_FF: True})

    @staticmethod
    def _converse_model(
        model_id: str = "anthropic.claude-sonnet-4-5-20250929-v1:0",
        response: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Mock]:
        reply = response or _converse_tool_call(END_EXECUTION_TOOL.name, {})
        client = Mock()
        client.converse.side_effect = lambda **_: copy.deepcopy(reply)
        model: Any = _StubChatBedrockConverse(
            model_id=model_id,
            region_name="us-east-1",
            client=client,
            bedrock_client=Mock(),
            **kwargs,
        )
        return model, client

    @staticmethod
    def _lookup_tool() -> BaseTool:
        return StructuredTool(
            name="lookup",
            description="Look something up.",
            args_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            func=lambda **_: None,
        )

    @staticmethod
    def _state(*messages: BaseMessage) -> AgentGraphState:
        return AgentGraphState(
            messages=[SystemMessage(content="You are an agent."), *messages]
        )

    @pytest.mark.asyncio
    async def test_forced_turn_marks_tools_system_and_last_message(self) -> None:
        self._enable()
        model, client = self._converse_model()

        node = create_llm_node(model, [self._lookup_tool()])
        assert isinstance(node, ForcedToolChoiceLLMNode)
        await node(self._state(HumanMessage(content="q")))

        assert _cache_points(client.converse.call_args.kwargs) == _ALL_MARKED

    @pytest.mark.asyncio
    async def test_flag_off_sends_no_cache_points(self) -> None:
        model, client = self._converse_model()

        await create_llm_node(model, [self._lookup_tool()])(
            self._state(HumanMessage(content="q"))
        )

        assert _cache_points(client.converse.call_args.kwargs) == _NONE_MARKED

    @pytest.mark.asyncio
    async def test_forced_extraction_call_sends_no_cache_points(self) -> None:
        """The extraction retry drops thinking and appends a nudge, so its message
        prefix differs from every cached turn; a cache write there is never read."""
        self._enable()
        model, client = self._converse_model(
            additional_model_request_fields={
                "thinking": {"type": "enabled", "budget_tokens": 1024}
            },
            supports_tool_choice_values=("auto", "any", "tool"),
        )

        await create_llm_node(model, [self._lookup_tool()])(
            self._state(HumanMessage(content="q"), AIMessage(content="an answer"))
        )

        request = client.converse.call_args.kwargs
        assert "thinking" not in request.get("additionalModelRequestFields", {})
        assert _cache_points(request) == _NONE_MARKED

    @pytest.mark.asyncio
    async def test_auto_tool_choice_marks_the_turn_but_not_the_finish_call(
        self,
    ) -> None:
        """The finish call binds only end_execution and raise_error, so even the
        tools segment differs from the run's cache entries."""
        self._enable()
        model, client = self._converse_model()
        model.model_details = _REJECTS_FORCING
        client.converse.side_effect = [
            _converse_text("Lisbon is warmer."),
            _converse_tool_call(END_EXECUTION_TOOL.name, {"answer": "Lisbon"}),
        ]
        tools = [self._lookup_tool(), *create_flow_control_tools(_Output)]

        node = create_llm_node(model, tools)
        assert isinstance(node, AutoToolChoiceLLMNode)
        await node(self._state(HumanMessage(content="q")))

        turn, finish = client.converse.call_args_list
        assert _cache_points(turn.kwargs) == _ALL_MARKED
        assert _cache_points(finish.kwargs) == _NONE_MARKED

    @pytest.mark.asyncio
    async def test_conversational_turn_is_marked(self) -> None:
        self._enable()
        model, client = self._converse_model(response=_converse_text("Hello!"))

        node = create_llm_node(model, [self._lookup_tool()], is_conversational=True)
        assert isinstance(node, ConversationalLLMNode)
        await node(self._state(HumanMessage(content="hi")))

        assert _cache_points(client.converse.call_args.kwargs) == _ALL_MARKED

    @pytest.mark.asyncio
    async def test_agent_without_tools_marks_system_and_last_message(self) -> None:
        self._enable()
        model, client = self._converse_model(response=_converse_text("Hello!"))

        await create_llm_node(model, [])(self._state(HumanMessage(content="hi")))

        assert _cache_points(client.converse.call_args.kwargs) == {
            "tools": 0,
            "system": 1,
            "earlier_messages": 0,
            "last_message": 1,
        }

    @pytest.mark.asyncio
    async def test_byo_connection_sends_no_cache_points(self) -> None:
        """Custom BYO gateways can rewrite the Converse body, and the rollout
        targets UiPath-billed traffic."""
        self._enable()
        model, client = self._converse_model()
        model.byo_connection_id = "6f1d2c3b-0000-0000-0000-000000000001"

        await create_llm_node(model, [self._lookup_tool()])(
            self._state(HumanMessage(content="q"))
        )

        assert _cache_points(client.converse.call_args.kwargs) == _NONE_MARKED

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_id",
        ["meta.llama3-1-70b-instruct-v1:0", "mistral.mistral-large-2407-v1:0"],
    )
    async def test_models_without_prompt_caching_send_no_cache_points(
        self, model_id: str
    ) -> None:
        self._enable()
        model, client = self._converse_model(
            model_id=model_id, supports_tool_choice_values=("auto", "any", "tool")
        )

        await create_llm_node(model, [self._lookup_tool()])(
            self._state(HumanMessage(content="q"))
        )

        assert _cache_points(client.converse.call_args.kwargs) == _NONE_MARKED

    @pytest.mark.asyncio
    async def test_non_bedrock_model_gets_no_cache_control(self) -> None:
        self._enable()
        model: Any = _StubAzureChatOpenAI.model_construct()
        model.bind_tools = Mock(return_value=model)
        model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="",
                tool_calls=[
                    create_tool_call(name=END_EXECUTION_TOOL.name, args={}, id="c1")
                ],
            )
        )

        await create_llm_node(model, [self._lookup_tool()])(
            self._state(HumanMessage(content="q"))
        )

        assert "cache_control" not in model.bind_tools.call_args.kwargs

    @pytest.mark.asyncio
    async def test_streamed_turn_carries_cache_points(self) -> None:
        self._enable()
        model, client = self._converse_model()
        client.converse_stream.return_value = _converse_stream_tool_call(
            END_EXECUTION_TOOL.name
        )
        graph = StateGraph(AgentGraphState)
        graph.add_node("llm", create_llm_node(model, [self._lookup_tool()]))
        graph.add_edge(START, "llm")
        graph.add_edge("llm", END)

        async for _ in graph.compile().astream(
            self._state(HumanMessage(content="q")), stream_mode="messages"
        ):
            pass

        client.converse.assert_not_called()
        assert _cache_points(client.converse_stream.call_args.kwargs) == _ALL_MARKED

    @pytest.mark.asyncio
    async def test_later_turns_keep_exactly_three_cache_points(self) -> None:
        """Bedrock allows 4 checkpoints per request; markers must not pile up in
        the run's messages across turns."""
        self._enable()
        model, client = self._converse_model(
            response=_converse_tool_call("lookup", {"q": "x"})
        )
        node = create_llm_node(model, [self._lookup_tool()])
        messages: list[BaseMessage] = [HumanMessage(content="q")]

        for turn in range(3):
            result = await node(self._state(*messages))
            reply = result["messages"][0]
            messages += [
                reply,
                ToolMessage(content=f"result {turn}", tool_call_id="c1"),
            ]

        for call in client.converse.call_args_list:
            assert _cache_points(call.kwargs) == _ALL_MARKED
        assert all(
            not isinstance(m.content, list)
            or not any(isinstance(b, dict) and "cachePoint" in b for b in m.content)
            for m in messages
        )
