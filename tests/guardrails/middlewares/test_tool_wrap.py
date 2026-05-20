"""Unit tests for BuiltInGuardrailMiddlewareMixin tool wrap logic.

Tests _extract_tool_input_data, _extract_tool_output_data, and _run_tool_guardrail
directly on UiPathPIIDetectionMiddleware (concrete subclass) with mocked guardrail
evaluation to avoid real UiPath API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import GuardrailScope
from uipath.platform.guardrails.decorators import LogAction
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.guardrails.enums import (
    GuardrailExecutionStage,
    PIIDetectionEntityType,
)
from uipath_langchain.guardrails.middlewares import UiPathPIIDetectionMiddleware
from uipath_langchain.guardrails.models import PIIDetectionEntity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOG = LogAction()

_PASSED = GuardrailValidationResult(result=GuardrailValidationResultType.PASSED, reason="")
_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="violation"
)


def _make_middleware(
    tools: list[str] | None = None,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
) -> UiPathPIIDetectionMiddleware:
    if tools is None:
        tools = ["my_tool"]
    return UiPathPIIDetectionMiddleware(
        scopes=[GuardrailScope.TOOL],
        action=_LOG,
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
        tools=tools,
        stage=stage,
    )


def _make_request(name: str = "my_tool", args: dict | str = None) -> ToolCallRequest:
    if args is None:
        args = {"text": "hello"}
    return ToolCallRequest(
        tool_call={"id": "tc1", "name": name, "args": args},
        tool=MagicMock(),
        state={},
        runtime=MagicMock(),
    )


def _make_tool_message(content: str = '{"result": "ok"}') -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="tc1")


# ---------------------------------------------------------------------------
# TestExtractToolInputData
# ---------------------------------------------------------------------------


class TestExtractToolInputData:
    def test_dict_args_returned_as_is(self) -> None:
        mw = _make_middleware()
        request = _make_request(args={"joke": "why did the chicken"})
        result = mw._extract_tool_input_data(request)
        assert result == {"joke": "why did the chicken"}

    def test_non_dict_args_stringified(self) -> None:
        mw = _make_middleware()
        request = _make_request(args="raw string arg")
        result = mw._extract_tool_input_data(request)
        assert result == "raw string arg"


# ---------------------------------------------------------------------------
# TestExtractToolOutputData
# ---------------------------------------------------------------------------


class TestExtractToolOutputData:
    def test_tool_message_json_dict(self) -> None:
        mw = _make_middleware()
        msg = _make_tool_message('{"key": "value"}')
        result = mw._extract_tool_output_data(msg)
        assert result == {"key": "value"}

    def test_tool_message_dict_content(self) -> None:
        mw = _make_middleware()
        msg = ToolMessage(content={"key": "value"}, tool_call_id="tc1")
        result = mw._extract_tool_output_data(msg)
        assert result == {"key": "value"}

    def test_tool_message_plain_string(self) -> None:
        mw = _make_middleware()
        msg = _make_tool_message("plain text output")
        result = mw._extract_tool_output_data(msg)
        assert result == {"output": "plain text output"}

    def test_tool_message_json_array(self) -> None:
        mw = _make_middleware()
        msg = _make_tool_message("[1, 2, 3]")
        result = mw._extract_tool_output_data(msg)
        assert result == {"output": [1, 2, 3]}

    def test_tool_message_ast_fallback(self) -> None:
        mw = _make_middleware()
        # Python literal — not valid JSON but parseable by ast.literal_eval
        msg = _make_tool_message("{'key': 'val'}")
        result = mw._extract_tool_output_data(msg)
        assert result == {"key": "val"}

    def test_command_no_tool_message_in_messages(self) -> None:
        mw = _make_middleware()
        # Command with a non-ToolMessage — content resolves to None → {}
        from langchain_core.messages import AIMessage

        cmd = Command(update={"messages": [AIMessage(content="hi")]})
        result = mw._extract_tool_output_data(cmd)
        assert result == {}

    def test_command_with_tool_message(self) -> None:
        mw = _make_middleware()
        tool_msg = _make_tool_message('{"cmd_result": "done"}')
        cmd = Command(update={"messages": [tool_msg]})
        result = mw._extract_tool_output_data(cmd)
        assert result == {"cmd_result": "done"}

    def test_command_without_messages(self) -> None:
        mw = _make_middleware()
        cmd = Command(update={"messages": []})
        result = mw._extract_tool_output_data(cmd)
        assert result == {}

    def test_unknown_type_returns_empty(self) -> None:
        mw = _make_middleware()
        result = mw._extract_tool_output_data("not a ToolMessage or Command")  # type: ignore[arg-type]
        assert result == {}


# ---------------------------------------------------------------------------
# TestRunToolGuardrail
# ---------------------------------------------------------------------------


class TestRunToolGuardrail:
    @pytest.mark.asyncio
    async def test_tool_not_in_tool_names_skips_guardrail(self) -> None:
        mw = _make_middleware(tools=["other_tool"])
        request = _make_request(name="my_tool")
        mock_handler = AsyncMock(return_value=_make_tool_message())

        with patch.object(mw, "_evaluate_guardrail") as mock_eval:
            result = await mw._run_tool_guardrail(request, mock_handler)

        mock_eval.assert_not_called()
        mock_handler.assert_called_once_with(request)
        assert isinstance(result, ToolMessage)

    @pytest.mark.asyncio
    async def test_tool_names_none_skips_guardrail(self) -> None:
        mw = _make_middleware()
        mw._tool_names = None  # force None to exercise the "is None" branch
        request = _make_request(name="my_tool")
        mock_handler = AsyncMock(return_value=_make_tool_message())

        with patch.object(mw, "_evaluate_guardrail") as mock_eval:
            await mw._run_tool_guardrail(request, mock_handler)

        mock_eval.assert_not_called()
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_pre_only_calls_evaluate_once_before_handler(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.PRE)
        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message())

        with patch.object(mw, "_evaluate_guardrail", return_value=_PASSED) as mock_eval:
            await mw._run_tool_guardrail(request, mock_handler)

        assert mock_eval.call_count == 1
        # PRE evaluation happens before handler — confirm handler was called
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_only_calls_evaluate_once_after_handler(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.POST)
        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message('{"x": 1}'))

        call_order: list[str] = []

        async def tracked_handler(req):
            call_order.append("handler")
            return _make_tool_message('{"x": 1}')

        mock_handler.side_effect = tracked_handler

        def tracked_eval(data):
            call_order.append("eval")
            return _PASSED

        with patch.object(mw, "_evaluate_guardrail", side_effect=tracked_eval):
            await mw._run_tool_guardrail(request, mock_handler)

        assert call_order == ["handler", "eval"], f"Expected handler then eval, got: {call_order}"

    @pytest.mark.asyncio
    async def test_pre_and_post_calls_evaluate_twice(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.PRE_AND_POST)
        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message('{"x": 1}'))

        with patch.object(mw, "_evaluate_guardrail", return_value=_PASSED) as mock_eval:
            await mw._run_tool_guardrail(request, mock_handler)

        assert mock_eval.call_count == 2

    @pytest.mark.asyncio
    async def test_pre_block_exception_converted_to_agent_error(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.PRE)
        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message())
        block_exc = GuardrailBlockException(title="Blocked", detail="PII detected")

        with patch.object(mw, "_evaluate_guardrail", side_effect=block_exc):
            with pytest.raises(AgentRuntimeError) as exc_info:
                await mw._run_tool_guardrail(request, mock_handler)

        assert exc_info.value.__cause__ is block_exc
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_agent_runtime_error_re_raised(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.PRE)
        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message())
        runtime_err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title="Test error",
            detail="Test detail",
        )

        with patch.object(mw, "_evaluate_guardrail", side_effect=runtime_err):
            with pytest.raises(AgentRuntimeError) as exc_info:
                await mw._run_tool_guardrail(request, mock_handler)

        assert exc_info.value is runtime_err
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_generic_exception_logged_not_raised(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.PRE)
        request = _make_request()
        expected_result = _make_tool_message()
        mock_handler = AsyncMock(return_value=expected_result)

        with patch.object(mw, "_evaluate_guardrail", side_effect=RuntimeError("oops")):
            result = await mw._run_tool_guardrail(request, mock_handler)

        # Handler still ran, result returned
        mock_handler.assert_called_once()
        assert result is expected_result

    @pytest.mark.asyncio
    async def test_pre_no_modification_when_none_returned(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.PRE)
        request = _make_request(args={"joke": "original"})
        mock_handler = AsyncMock(return_value=_make_tool_message())

        with patch.object(mw, "_evaluate_guardrail", return_value=_PASSED):
            await mw._run_tool_guardrail(request, mock_handler)

        # Handler should receive the original (unmodified) request
        called_request = mock_handler.call_args[0][0]
        assert called_request.tool_call["args"] == {"joke": "original"}

    @pytest.mark.asyncio
    async def test_pre_modifies_request_when_dict_returned(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.PRE)
        mw.action = MagicMock()
        mw.action.handle_validation_result.return_value = {"joke": "filtered"}

        request = _make_request(args={"joke": "original"})
        mock_handler = AsyncMock(return_value=_make_tool_message())

        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            await mw._run_tool_guardrail(request, mock_handler)

        called_request = mock_handler.call_args[0][0]
        assert called_request.tool_call["args"] == {"joke": "filtered"}

    @pytest.mark.asyncio
    async def test_post_skips_when_empty_output_data(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.POST)
        request = _make_request()
        # Command with no ToolMessage → _extract_tool_output_data returns {}
        mock_handler = AsyncMock(return_value=Command(update={"messages": []}))

        call_count = 0

        def counting_eval(data):
            nonlocal call_count
            call_count += 1
            return _PASSED

        with patch.object(mw, "_evaluate_guardrail", side_effect=counting_eval):
            await mw._run_tool_guardrail(request, mock_handler)

        assert call_count == 0, "evaluate_guardrail should not be called when output is empty"

    @pytest.mark.asyncio
    async def test_post_block_exception_converted_to_agent_error(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.POST)
        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message('{"x": 1}'))
        block_exc = GuardrailBlockException(title="Blocked", detail="PII in output")

        with patch.object(mw, "_evaluate_guardrail", side_effect=block_exc):
            with pytest.raises(AgentRuntimeError) as exc_info:
                await mw._run_tool_guardrail(request, mock_handler)

        assert exc_info.value.__cause__ is block_exc

    @pytest.mark.asyncio
    async def test_post_agent_runtime_error_re_raised(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.POST)
        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message('{"x": 1}'))
        runtime_err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title="Test error",
            detail="Test detail",
        )

        with patch.object(mw, "_evaluate_guardrail", side_effect=runtime_err):
            with pytest.raises(AgentRuntimeError) as exc_info:
                await mw._run_tool_guardrail(request, mock_handler)

        assert exc_info.value is runtime_err

    @pytest.mark.asyncio
    async def test_post_generic_exception_logged_not_raised(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.POST)
        request = _make_request()
        expected_result = _make_tool_message('{"x": 1}')
        mock_handler = AsyncMock(return_value=expected_result)

        with patch.object(mw, "_evaluate_guardrail", side_effect=RuntimeError("boom")):
            result = await mw._run_tool_guardrail(request, mock_handler)

        # Result unchanged when POST raises generic exception
        assert result is expected_result

    @pytest.mark.asyncio
    async def test_post_no_modification_when_none_returned(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.POST)
        request = _make_request()
        original_result = _make_tool_message('{"x": 1}')
        mock_handler = AsyncMock(return_value=original_result)

        with patch.object(mw, "_evaluate_guardrail", return_value=_PASSED):
            result = await mw._run_tool_guardrail(request, mock_handler)

        assert result is original_result

    @pytest.mark.asyncio
    async def test_post_modifies_result_when_output_returned(self) -> None:
        mw = _make_middleware(stage=GuardrailExecutionStage.POST)
        mw.action = MagicMock()
        mw.action.handle_validation_result.return_value = {"x": "filtered"}

        request = _make_request()
        mock_handler = AsyncMock(return_value=_make_tool_message('{"x": 1}'))

        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            result = await mw._run_tool_guardrail(request, mock_handler)

        assert isinstance(result, ToolMessage)
        assert "filtered" in result.content
