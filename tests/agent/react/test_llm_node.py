"""Tests for LLM node tool call filtering functionality."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.content import create_text_block, create_tool_call
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.llm_client import UiPathAPIError, UiPathError, UiPathLLMErrorCode
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.llm_node import create_llm_node
from uipath_langchain.agent.react.types import AgentGraphState


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
        request = httpx.Request("POST", "http://gateway/")
        return httpx.Response(
            403, request=request, json={"status": 403, "detail": "need AGU"}
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
        model = _StubAzureChatOpenAI.model_construct()
        model.thinking = {
            "type": "adaptive"
        }  # thinking active; OpenAI MRO won't downgrade
        model.bind_tools = Mock(return_value=model)
        model.bind = Mock(return_value=model)
        return model

    def _plain_model(self) -> Any:
        model = _StubAzureChatOpenAI.model_construct()
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

    async def _run_capture(self, model: Any, tool_choice: str = "auto") -> list[Any]:
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
