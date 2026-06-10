"""Tests that the middleware mixin publishes guardrail context to the action
and re-raises LangGraph control-flow signals (so interrupt() is not swallowed).
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.prebuilt.tool_node import ToolCallRequest
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)

from uipath_langchain.guardrails import GuardrailAction
from uipath_langchain.guardrails._action_context import (
    GuardrailActionContext,
    current_action_context,
)
from uipath_langchain.guardrails.enums import (
    GuardrailExecutionStage,
    PIIDetectionEntityType,
)
from uipath_langchain.guardrails.middlewares import UiPathPIIDetectionMiddleware
from uipath_langchain.guardrails.models import PIIDetectionEntity

_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="violation"
)
_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)


class _RecordingAction(GuardrailAction):
    """Captures the published context when invoked; optionally raises."""

    def __init__(self, raise_exc: BaseException | None = None) -> None:
        self.seen: GuardrailActionContext | None = None
        self.called = False
        self._raise = raise_exc

    def handle_validation_result(
        self, result: Any, data: Any, guardrail_name: str
    ) -> Any:
        self.called = True
        self.seen = current_action_context()
        if self._raise is not None:
            raise self._raise
        return None


def _message_mw(
    action: GuardrailAction, scope: GuardrailScope = GuardrailScope.AGENT
) -> UiPathPIIDetectionMiddleware:
    return UiPathPIIDetectionMiddleware(
        scopes=[scope],
        action=action,
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
    )


def _tool_mw(
    action: GuardrailAction,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE,
) -> UiPathPIIDetectionMiddleware:
    return UiPathPIIDetectionMiddleware(
        scopes=[GuardrailScope.TOOL],
        action=action,
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
        tools=["my_tool"],
        stage=stage,
    )


def _request() -> ToolCallRequest:
    tool_call: Any = {"id": "tc1", "name": "my_tool", "args": {"text": "a@b.com"}}
    return ToolCallRequest(
        tool_call=tool_call, tool=MagicMock(), state={}, runtime=MagicMock()
    )


# ---------------------------------------------------------------------------
# Context publishing — message scopes (_check_messages)
# ---------------------------------------------------------------------------


class TestMessageContextPublishing:
    def test_agent_scope_publishes_agent_pre_context(self) -> None:
        action = _RecordingAction()
        mw = _message_mw(action, scope=GuardrailScope.AGENT)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            mw._check_messages(
                [HumanMessage(content="hi a@b.com")],
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.PRE,
            )
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.AGENT
        assert action.seen.execution_stage == GuardrailExecutionStage.PRE
        assert action.seen.component == "Agent"

    def test_llm_scope_publishes_llm_call_component(self) -> None:
        action = _RecordingAction()
        mw = _message_mw(action, scope=GuardrailScope.LLM)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            mw._check_messages(
                [HumanMessage(content="hi a@b.com")],
                scope=GuardrailScope.LLM,
                stage=GuardrailExecutionStage.POST,
            )
        assert action.seen is not None
        assert action.seen.component == "LLM call"
        assert action.seen.execution_stage == GuardrailExecutionStage.POST

    def test_context_reset_after_call(self) -> None:
        action = _RecordingAction()
        mw = _message_mw(action)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            mw._check_messages(
                [HumanMessage(content="hi a@b.com")],
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.PRE,
            )
        assert current_action_context() is None

    def test_passed_result_does_not_invoke_action(self) -> None:
        action = _RecordingAction()
        mw = _message_mw(action)
        with patch.object(mw, "_evaluate_guardrail", return_value=_PASSED):
            mw._check_messages(
                [HumanMessage(content="clean")],
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.PRE,
            )
        assert action.called is False

    def test_guardrail_description_is_published(self) -> None:
        action = _RecordingAction()
        mw = _message_mw(action, scope=GuardrailScope.AGENT)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            mw._check_messages(
                [HumanMessage(content="hi a@b.com")],
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.PRE,
            )
        assert action.seen is not None
        assert action.seen.description == mw._guardrail.description

    def test_post_publishes_input_text_as_input_payload(self) -> None:
        action = _RecordingAction()
        mw = _message_mw(action, scope=GuardrailScope.AGENT)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            mw._check_messages(
                [AIMessage(content="output a@b.com")],
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.POST,
                input_text="the original input",
            )
        assert action.seen is not None
        assert action.seen.input_payload == json.dumps("the original input")


# ---------------------------------------------------------------------------
# Context publishing — tool scope (_run_tool_guardrail)
# ---------------------------------------------------------------------------


class TestToolContextPublishing:
    @pytest.mark.asyncio
    async def test_pre_publishes_tool_pre_context_with_tool_name(self) -> None:
        action = _RecordingAction()
        mw = _tool_mw(action, stage=GuardrailExecutionStage.PRE)
        handler = AsyncMock(return_value=ToolMessage(content="{}", tool_call_id="tc1"))
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            await mw._run_tool_guardrail(_request(), handler)
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.TOOL
        assert action.seen.execution_stage == GuardrailExecutionStage.PRE
        assert action.seen.component == "my_tool"
        assert action.seen.input_payload is None  # no separate input at PRE

    @pytest.mark.asyncio
    async def test_post_publishes_tool_post_context(self) -> None:
        action = _RecordingAction()
        mw = _tool_mw(action, stage=GuardrailExecutionStage.POST)
        handler = AsyncMock(
            return_value=ToolMessage(content='{"x": 1}', tool_call_id="tc1")
        )
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            await mw._run_tool_guardrail(_request(), handler)
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.TOOL
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.component == "my_tool"
        assert action.seen.input_payload == json.dumps({"text": "a@b.com"})


# ---------------------------------------------------------------------------
# Bubble-up: interrupt() control-flow signal must NOT be swallowed
# ---------------------------------------------------------------------------


class TestGraphBubbleUpReraised:
    def test_check_messages_reraises_graph_interrupt(self) -> None:
        action = _RecordingAction(raise_exc=GraphInterrupt(()))
        mw = _message_mw(action)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            with pytest.raises(GraphInterrupt):
                mw._check_messages(
                    [HumanMessage(content="a@b.com")],
                    scope=GuardrailScope.AGENT,
                    stage=GuardrailExecutionStage.PRE,
                )

    def test_check_messages_still_swallows_generic_exception(self) -> None:
        action = _RecordingAction(raise_exc=RuntimeError("boom"))
        mw = _message_mw(action)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            # Generic errors stay swallowed/logged (no raise) — regression guard.
            mw._check_messages(
                [HumanMessage(content="a@b.com")],
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.PRE,
            )

    @pytest.mark.asyncio
    async def test_run_tool_guardrail_reraises_graph_interrupt(self) -> None:
        action = _RecordingAction(raise_exc=GraphInterrupt(()))
        mw = _tool_mw(action, stage=GuardrailExecutionStage.PRE)
        handler = AsyncMock(return_value=ToolMessage(content="{}", tool_call_id="tc1"))
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            with pytest.raises(GraphInterrupt):
                await mw._run_tool_guardrail(_request(), handler)
