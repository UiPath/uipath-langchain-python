"""Unit tests for UiPathLLMAsJudgeMiddleware.

Pure unit tests: the guardrail is built offline (no network) and
``_evaluate_guardrail`` is always patched, so no call reaches
``/agentsruntime_/api/execution/guardrails/validate``.
"""

from typing import Any, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import GuardrailScope
from uipath.platform.guardrails.decorators import BlockAction, LogAction
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.guardrails.enums import GuardrailExecutionStage
from uipath_langchain.guardrails.middlewares import UiPathLLMAsJudgeMiddleware

_LOG = LogAction()
_BLOCK = BlockAction()

_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)
_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="violation"
)

_RULE = "The answer must be genuinely funny, clean, and on-topic."


def _hook_names(middleware: Iterable[Any]) -> list[str]:
    return [inst.name for inst in middleware]


def _make_mw(
    *,
    scopes: list[GuardrailScope] | None = None,
    action: Any = _LOG,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
    tools: list[str] | None = None,
    guardrail_text: str = _RULE,
    model: str = "gpt-4o-2024-08-06",
    positive_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
    threshold: float = 2.0,
) -> UiPathLLMAsJudgeMiddleware:
    return UiPathLLMAsJudgeMiddleware(
        scopes=scopes or [GuardrailScope.AGENT],
        action=action,
        guardrail_text=guardrail_text,
        model=model,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        threshold=threshold,
        stage=stage,
        tools=tools,
    )


def _make_tool_request(
    name: str = "my_tool", args: dict[str, Any] | None = None
) -> ToolCallRequest:
    tool_call: Any = {"id": "tc1", "name": name, "args": args or {"joke": "hi"}}
    return ToolCallRequest(
        tool_call=tool_call, tool=MagicMock(), state={}, runtime=MagicMock()
    )


# --- (a) builds the right BuiltInValidatorGuardrail -------------------------
class TestGuardrailConstruction:
    def test_validator_type_and_kind(self) -> None:
        g = _make_mw()._guardrail
        assert g.guardrail_type == "builtInValidator"
        assert g.validator_type == "llm_as_judge"

    def test_required_parameters(self) -> None:
        params = {p.id: p for p in _make_mw()._guardrail.validator_parameters}
        assert params["guardrailText"].parameter_type == "text"
        assert params["guardrailText"].value == _RULE
        assert params["model"].parameter_type == "enum"
        assert params["model"].value == "gpt-4o-2024-08-06"
        assert params["threshold"].parameter_type == "number"
        assert params["threshold"].value == 2.0

    def test_example_parameters_included_only_when_present(self) -> None:
        # No examples passed -> no text-list params.
        ids = {p.id for p in _make_mw()._guardrail.validator_parameters}
        assert "positiveExamples" not in ids
        assert "negativeExamples" not in ids

        params = {
            p.id: p
            for p in _make_mw(
                positive_examples=["a clean pun"],
                negative_examples=["not a joke"],
            )._guardrail.validator_parameters
        }
        assert params["positiveExamples"].parameter_type == "text-list"
        assert params["positiveExamples"].value == ["a clean pun"]
        assert params["negativeExamples"].value == ["not a joke"]

    def test_selector_scopes_and_tool_match_names(self) -> None:
        selector = _make_mw(scopes=[GuardrailScope.LLM])._guardrail.selector
        assert selector is not None
        assert GuardrailScope.LLM in selector.scopes

        tool_selector = _make_mw(
            scopes=[GuardrailScope.TOOL], tools=["analyze_joke_syntax"]
        )._guardrail.selector
        assert tool_selector is not None
        assert tool_selector.match_names == ["analyze_joke_syntax"]


# --- (b) hooks fire at the configured scope/stage ---------------------------
class TestHookWiring:
    def test_agent_pre_registers_only_before_agent(self) -> None:
        names = _hook_names(_make_mw(stage=GuardrailExecutionStage.PRE))
        assert names == ["LLM_as_Judge_before_agent"]

    def test_agent_post_registers_only_after_agent(self) -> None:
        names = _hook_names(_make_mw(stage=GuardrailExecutionStage.POST))
        assert names == ["LLM_as_Judge_after_agent"]

    def test_llm_pre_and_post_registers_both(self) -> None:
        names = _hook_names(
            _make_mw(
                scopes=[GuardrailScope.LLM],
                stage=GuardrailExecutionStage.PRE_AND_POST,
            )
        )
        assert sorted(names) == [
            "LLM_as_Judge_after_model",
            "LLM_as_Judge_before_model",
        ]

    def test_tool_scope_registers_single_wrap_hook(self) -> None:
        names = _hook_names(
            _make_mw(scopes=[GuardrailScope.TOOL], tools=["analyze_joke_syntax"])
        )
        assert len(names) == 1
        assert "wrap_tool_call" in names[0]

    def test_all_scopes_register_all_hooks(self) -> None:
        names = _hook_names(
            _make_mw(
                scopes=[
                    GuardrailScope.AGENT,
                    GuardrailScope.LLM,
                    GuardrailScope.TOOL,
                ],
                tools=["analyze_joke_syntax"],
                stage=GuardrailExecutionStage.PRE_AND_POST,
            )
        )
        assert sorted(names) == [
            "LLM_as_Judge_after_agent",
            "LLM_as_Judge_after_model",
            "LLM_as_Judge_before_agent",
            "LLM_as_Judge_before_model",
            "LLM_as_Judge_wrap_tool_call",
        ]


# --- (c) message actions on a mocked verdict --------------------------------
class TestMessageActions:
    def test_passed_does_not_invoke_action(self) -> None:
        mw = _make_mw()
        mw.action = MagicMock()
        with patch.object(mw, "_evaluate_guardrail", return_value=_PASSED):
            mw._check_messages(
                [HumanMessage(content="hello")],
                scope=GuardrailScope.AGENT,
                stage=GuardrailExecutionStage.PRE,
            )
        mw.action.handle_validation_result.assert_not_called()

    def test_log_action_leaves_message_unchanged(self) -> None:
        mw = _make_mw(action=_LOG)
        msg = HumanMessage(content="risky content")
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            mw._check_messages(
                [msg], scope=GuardrailScope.AGENT, stage=GuardrailExecutionStage.PRE
            )
        assert msg.content == "risky content"

    def test_filter_action_modifies_message_content(self) -> None:
        mw = _make_mw()
        mw.action = MagicMock()
        mw.action.handle_validation_result.return_value = "[REDACTED]"
        msg = HumanMessage(content="risky content")
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            mw._check_messages(
                [msg], scope=GuardrailScope.AGENT, stage=GuardrailExecutionStage.PRE
            )
        assert msg.content == "[REDACTED]"

    def test_block_action_raises_agent_runtime_error(self) -> None:
        mw = _make_mw(action=_BLOCK)
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            with pytest.raises(AgentRuntimeError) as exc_info:
                mw._check_messages(
                    [HumanMessage(content="risky content")],
                    scope=GuardrailScope.AGENT,
                    stage=GuardrailExecutionStage.PRE,
                )
        assert isinstance(exc_info.value.__cause__, GuardrailBlockException)


# --- (d) TOOL async path ----------------------------------------------------
class TestToolPath:
    async def test_tool_block_raises_and_skips_handler(self) -> None:
        mw = _make_mw(
            scopes=[GuardrailScope.TOOL],
            action=_BLOCK,
            tools=["my_tool"],
            stage=GuardrailExecutionStage.PRE,
        )
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="tc1"))
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            with pytest.raises(AgentRuntimeError):
                await mw._run_tool_guardrail(_make_tool_request(), handler)
        handler.assert_not_called()

    async def test_tool_filter_modifies_request_args(self) -> None:
        mw = _make_mw(
            scopes=[GuardrailScope.TOOL],
            tools=["my_tool"],
            stage=GuardrailExecutionStage.PRE,
        )
        mw.action = MagicMock()
        mw.action.handle_validation_result.return_value = {"joke": "filtered"}
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="tc1"))
        with patch.object(mw, "_evaluate_guardrail", return_value=_FAILED):
            await mw._run_tool_guardrail(_make_tool_request(), handler)
        assert handler.call_args[0][0].tool_call["args"] == {"joke": "filtered"}


# --- Construction validation ------------------------------------------------
class TestValidation:
    def test_empty_guardrail_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="guardrail_text"):
            _make_mw(guardrail_text="  ")

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="model"):
            _make_mw(model="")

    def test_threshold_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="Threshold"):
            _make_mw(threshold=7.0)

    def test_tool_scope_without_tools_rejected(self) -> None:
        with pytest.raises(ValueError, match="Tool scope"):
            _make_mw(scopes=[GuardrailScope.TOOL])

    def test_guardrail_text_over_limit_rejected(self) -> None:
        with pytest.raises(ValueError, match="guardrail_text exceeds"):
            _make_mw(guardrail_text="x" * 4001)

    def test_guardrail_text_at_limit_ok(self) -> None:
        _make_mw(guardrail_text="x" * 4000)

    def test_too_many_positive_examples_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive_examples allows at most 2"):
            _make_mw(positive_examples=["a", "b", "c"])

    def test_too_many_negative_examples_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative_examples allows at most 2"):
            _make_mw(negative_examples=["a", "b", "c"])

    def test_two_examples_each_ok(self) -> None:
        _make_mw(positive_examples=["a", "b"], negative_examples=["c", "d"])

    def test_positive_example_over_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive_examples entry"):
            _make_mw(positive_examples=["x" * 1001])

    def test_negative_example_over_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative_examples entry"):
            _make_mw(negative_examples=["x" * 1001])

    def test_example_at_length_limit_ok(self) -> None:
        _make_mw(positive_examples=["x" * 1000])
