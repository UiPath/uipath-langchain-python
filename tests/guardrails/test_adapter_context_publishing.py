"""Tests that the LangChain adapter publishes guardrail context to the action.

The decorator (``@guardrail``) path wraps LangChain/LangGraph objects via
``LangChainGuardrailAdapter``. Like the middleware path, the adapter must publish
the guardrail runtime context (scope / stage / component) around each action call
so context-aware actions (e.g. ``EscalateAction``) can derive
``Component`` / ``ExecutionStage`` instead of requiring them to be hardcoded.

The action call sites only catch ``GuardrailBlockException`` (the broad
``except Exception`` is around the evaluator, not the action), so a LangGraph
``GraphInterrupt`` raised by an escalation action propagates unchanged — these
tests guard that too.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt
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
from uipath_langchain.guardrails._langchain_adapter import (
    LangChainGuardrailAdapter,
    _apply_agent_input_guardrail,
    _apply_agent_output_guardrail,
    _apply_llm_post,
    _apply_llm_pre,
    _input_payload_from_messages,
    _wrap_compiled_graph_with_guardrail,
    _wrap_llm_with_guardrail,
    _wrap_stategraph_with_guardrail,
    _wrap_tool_with_guardrail,
)
from uipath_langchain.guardrails.enums import GuardrailExecutionStage

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


def _fail_eval(*_args: Any, **_kwargs: Any) -> GuardrailValidationResult:
    return _FAILED


def _pass_eval(*_args: Any, **_kwargs: Any) -> GuardrailValidationResult:
    return _PASSED


def _raise_eval(*_args: Any, **_kwargs: Any) -> GuardrailValidationResult:
    """Evaluator that raises — exercises the ``except Exception`` log branches."""
    raise RuntimeError("evaluator boom")


class _ModifyingAction(GuardrailAction):
    """Action that returns a modified string — exercises the apply-modification branches."""

    def handle_validation_result(
        self, result: Any, data: Any, guardrail_name: str
    ) -> Any:
        return "MODIFIED"


@tool
def my_tool(text: str) -> str:
    """Echo the input text back."""
    return f"echo: {text}"


class _FakeChatModel(BaseChatModel):
    """Minimal BaseChatModel whose response is a fixed AIMessage.

    Used to drive the ``_wrap_llm_with_guardrail`` ``invoke``/``ainvoke`` wrappers
    (the wrapper swaps ``__class__``, exactly like the tool wrapper).
    """

    reply: str

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.reply))]
        )


class _FakeGraph:
    """Minimal graph-like object exposing ``invoke``/``ainvoke`` for the graph wrappers."""

    def __init__(self, output: dict[str, Any]) -> None:
        self._output = output

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._output

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._output


# ---------------------------------------------------------------------------
# Tool scope — context published via the wrapped tool's invoke()
# ---------------------------------------------------------------------------


class TestToolContextPublishing:
    def test_pre_publishes_tool_pre_context_with_tool_name(self) -> None:
        action = _RecordingAction()
        wrapped = _wrap_tool_with_guardrail(
            my_tool, _fail_eval, action, "g", GuardrailExecutionStage.PRE
        )
        wrapped.invoke({"text": "a@b.com"})
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.TOOL
        assert action.seen.execution_stage == GuardrailExecutionStage.PRE
        assert action.seen.component == "my_tool"
        assert action.seen.input_payload is None  # no separate input at PRE

    def test_post_publishes_tool_post_context(self) -> None:
        action = _RecordingAction()
        wrapped = _wrap_tool_with_guardrail(
            my_tool, _fail_eval, action, "g", GuardrailExecutionStage.POST
        )
        wrapped.invoke({"text": "a@b.com"})
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.TOOL
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.component == "my_tool"
        # POST carries the original tool input so the action shows both sides.
        assert action.seen.input_payload == json.dumps({"text": "a@b.com"})

    def test_context_reset_after_invoke(self) -> None:
        action = _RecordingAction()
        wrapped = _wrap_tool_with_guardrail(
            my_tool, _fail_eval, action, "g", GuardrailExecutionStage.PRE
        )
        wrapped.invoke({"text": "a@b.com"})
        assert current_action_context() is None

    def test_pre_reraises_graph_interrupt(self) -> None:
        action = _RecordingAction(raise_exc=GraphInterrupt(()))
        wrapped = _wrap_tool_with_guardrail(
            my_tool, _fail_eval, action, "g", GuardrailExecutionStage.PRE
        )
        with pytest.raises(GraphInterrupt):
            wrapped.invoke({"text": "a@b.com"})


# ---------------------------------------------------------------------------
# LLM scope — context published via _apply_llm_pre / _apply_llm_post
# ---------------------------------------------------------------------------


class TestLLMContextPublishing:
    def test_pre_publishes_llm_pre_context(self) -> None:
        action = _RecordingAction()
        _apply_llm_pre([HumanMessage(content="hi a@b.com")], _fail_eval, action, "g")
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.LLM
        assert action.seen.execution_stage == GuardrailExecutionStage.PRE
        assert action.seen.component == "LLM call"
        assert action.seen.input_payload is None  # no separate input at PRE

    def test_post_publishes_llm_post_context(self) -> None:
        action = _RecordingAction()
        _apply_llm_post(
            AIMessage(content="answer a@b.com"),
            _fail_eval,
            action,
            "g",
            input_messages=[HumanMessage(content="hi a@b.com")],
        )
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.LLM
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.component == "LLM call"
        # POST carries the original human input so the action shows both sides.
        assert action.seen.input_payload == json.dumps("hi a@b.com")

    def test_context_reset_after_call(self) -> None:
        action = _RecordingAction()
        _apply_llm_pre([HumanMessage(content="hi a@b.com")], _fail_eval, action, "g")
        assert current_action_context() is None

    def test_passed_result_does_not_invoke_action(self) -> None:
        action = _RecordingAction()
        _apply_llm_pre([HumanMessage(content="clean")], _pass_eval, action, "g")
        assert action.called is False

    def test_pre_reraises_graph_interrupt(self) -> None:
        action = _RecordingAction(raise_exc=GraphInterrupt(()))
        with pytest.raises(GraphInterrupt):
            _apply_llm_pre([HumanMessage(content="a@b.com")], _fail_eval, action, "g")


# ---------------------------------------------------------------------------
# Agent scope — context published via _apply_agent_input/output_guardrail
# ---------------------------------------------------------------------------


class TestAgentContextPublishing:
    def test_input_publishes_agent_pre_context(self) -> None:
        action = _RecordingAction()
        _apply_agent_input_guardrail(
            {"messages": [HumanMessage(content="hi a@b.com")]},
            _fail_eval,
            action,
            "g",
        )
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.AGENT
        assert action.seen.execution_stage == GuardrailExecutionStage.PRE
        assert action.seen.component == "Agent"
        assert action.seen.input_payload is None  # no separate input at PRE

    def test_output_publishes_agent_post_context(self) -> None:
        action = _RecordingAction()
        _apply_agent_output_guardrail(
            {
                "messages": [
                    HumanMessage(content="hi a@b.com"),
                    AIMessage(content="answer a@b.com"),
                ]
            },
            _fail_eval,
            action,
            "g",
        )
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.AGENT
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.component == "Agent"
        # POST carries the original human input (from the output messages).
        assert action.seen.input_payload == json.dumps("hi a@b.com")

    def test_context_reset_after_call(self) -> None:
        action = _RecordingAction()
        _apply_agent_input_guardrail(
            {"messages": [HumanMessage(content="hi a@b.com")]},
            _fail_eval,
            action,
            "g",
        )
        assert current_action_context() is None

    def test_input_reraises_graph_interrupt(self) -> None:
        action = _RecordingAction(raise_exc=GraphInterrupt(()))
        with pytest.raises(GraphInterrupt):
            _apply_agent_input_guardrail(
                {"messages": [HumanMessage(content="a@b.com")]},
                _fail_eval,
                action,
                "g",
            )


# ---------------------------------------------------------------------------
# _input_payload_from_messages — the original-input-at-POST helper
# ---------------------------------------------------------------------------


class TestInputPayloadFromMessages:
    def test_none_when_no_messages(self) -> None:
        assert _input_payload_from_messages(None) is None
        assert _input_payload_from_messages([]) is None

    def test_none_when_no_human_message(self) -> None:
        assert _input_payload_from_messages([AIMessage(content="x")]) is None

    def test_json_encodes_last_human_text(self) -> None:
        assert _input_payload_from_messages(
            [HumanMessage(content="hi there")]
        ) == json.dumps("hi there")


# ---------------------------------------------------------------------------
# LLM scope — context published via the wrapped model's invoke()/ainvoke()
# ---------------------------------------------------------------------------


class TestLLMWrapperContextPublishing:
    def test_sync_invoke_publishes_llm_context_with_input_payload(self) -> None:
        action = _RecordingAction()
        llm = _FakeChatModel(reply="answer a@b.com")
        wrapped = _wrap_llm_with_guardrail(
            llm, _fail_eval, action, "g", GuardrailExecutionStage.PRE_AND_POST
        )
        wrapped.invoke([HumanMessage(content="hi a@b.com")])
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.LLM
        assert action.seen.component == "LLM call"
        # POST fires last; it carries the original input alongside the output.
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.input_payload == json.dumps("hi a@b.com")

    @pytest.mark.asyncio
    async def test_async_ainvoke_publishes_llm_context_with_input_payload(self) -> None:
        action = _RecordingAction()
        llm = _FakeChatModel(reply="answer a@b.com")
        wrapped = _wrap_llm_with_guardrail(
            llm, _fail_eval, action, "g", GuardrailExecutionStage.PRE_AND_POST
        )
        await wrapped.ainvoke([HumanMessage(content="hi a@b.com")])
        assert action.seen is not None
        assert action.seen.component == "LLM call"
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.input_payload == json.dumps("hi a@b.com")


# ---------------------------------------------------------------------------
# Graph wrappers — StateGraph and CompiledStateGraph invoke()/ainvoke()
# ---------------------------------------------------------------------------


def _graph_output() -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content="hi a@b.com"),
            AIMessage(content="answer a@b.com"),
        ]
    }


class TestGraphWrapperContextPublishing:
    # PRE_AND_POST exercises both the PRE (input) and POST (output) branches in one
    # call; the recorded context is the last (POST) one.

    def test_stategraph_sync_invoke_publishes_agent_context(self) -> None:
        action = _RecordingAction()
        graph: Any = _FakeGraph(_graph_output())
        _wrap_stategraph_with_guardrail(
            graph, _fail_eval, action, "g", GuardrailExecutionStage.PRE_AND_POST
        )
        graph.invoke({"messages": [HumanMessage(content="hi a@b.com")]})
        assert action.seen is not None
        assert action.seen.scope == GuardrailScope.AGENT
        assert action.seen.component == "Agent"
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.input_payload == json.dumps("hi a@b.com")

    @pytest.mark.asyncio
    async def test_stategraph_async_ainvoke_publishes_agent_context(self) -> None:
        action = _RecordingAction()
        graph: Any = _FakeGraph(_graph_output())
        _wrap_stategraph_with_guardrail(
            graph, _fail_eval, action, "g", GuardrailExecutionStage.PRE_AND_POST
        )
        await graph.ainvoke({"messages": [HumanMessage(content="hi a@b.com")]})
        assert action.seen is not None
        assert action.seen.component == "Agent"
        assert action.seen.execution_stage == GuardrailExecutionStage.POST

    def test_compiled_graph_sync_invoke_publishes_agent_context(self) -> None:
        action = _RecordingAction()
        graph: Any = _FakeGraph(_graph_output())
        _wrap_compiled_graph_with_guardrail(
            graph, _fail_eval, action, "g", GuardrailExecutionStage.PRE_AND_POST
        )
        graph.invoke({"messages": [HumanMessage(content="hi a@b.com")]})
        assert action.seen is not None
        assert action.seen.component == "Agent"
        assert action.seen.execution_stage == GuardrailExecutionStage.POST
        assert action.seen.input_payload == json.dumps("hi a@b.com")

    @pytest.mark.asyncio
    async def test_compiled_graph_async_ainvoke_publishes_agent_context(self) -> None:
        action = _RecordingAction()
        graph: Any = _FakeGraph(_graph_output())
        _wrap_compiled_graph_with_guardrail(
            graph, _fail_eval, action, "g", GuardrailExecutionStage.PRE_AND_POST
        )
        await graph.ainvoke({"messages": [HumanMessage(content="hi a@b.com")]})
        assert action.seen is not None
        assert action.seen.component == "Agent"
        assert action.seen.execution_stage == GuardrailExecutionStage.POST


# ---------------------------------------------------------------------------
# Defensive branches — evaluator errors are swallowed; malformed input no-ops
# ---------------------------------------------------------------------------


class TestAdapterDefensiveBranches:
    def test_tool_pre_swallows_evaluator_exception(self) -> None:
        # Fresh tool instance: _wrap_tool_with_guardrail swaps __class__, and the
        # module-level my_tool is shared, so wrapping a fresh tool avoids stacking.
        @tool
        def echo_pre(text: str) -> str:
            """Echo the input text back."""
            return f"echo: {text}"

        action = _RecordingAction()
        wrapped = _wrap_tool_with_guardrail(
            echo_pre, _raise_eval, action, "g", GuardrailExecutionStage.PRE
        )
        # Evaluator raises → logged & swallowed; tool still runs, action not called.
        assert wrapped.invoke({"text": "x"}) == "echo: x"
        assert action.called is False

    def test_tool_post_swallows_evaluator_exception(self) -> None:
        @tool
        def echo_post(text: str) -> str:
            """Echo the input text back."""
            return f"echo: {text}"

        action = _RecordingAction()
        wrapped = _wrap_tool_with_guardrail(
            echo_post, _raise_eval, action, "g", GuardrailExecutionStage.POST
        )
        assert wrapped.invoke({"text": "x"}) == "echo: x"
        assert action.called is False

    @pytest.mark.parametrize(
        "bad_input",
        ["not a dict", {}, {"messages": "notalist"}, {"messages": []}],
    )
    def test_agent_input_guardrail_noops_on_malformed_input(
        self, bad_input: Any
    ) -> None:
        action = _RecordingAction()
        _apply_agent_input_guardrail(bad_input, _fail_eval, action, "g")
        assert action.called is False

    @pytest.mark.parametrize(
        "bad_output",
        [
            "not a dict",
            {},
            {"messages": "notalist"},
            {"messages": [HumanMessage(content="only human")]},
            {"messages": [AIMessage(content="")]},
        ],
    )
    def test_agent_output_guardrail_noops_on_malformed_output(
        self, bad_output: Any
    ) -> None:
        action = _RecordingAction()
        _apply_agent_output_guardrail(bad_output, _fail_eval, action, "g")
        assert action.called is False


# ---------------------------------------------------------------------------
# Apply helpers — empty/raise no-ops and modification application
# ---------------------------------------------------------------------------


class TestApplyHelperBranches:
    # --- LLM pre ---
    def test_llm_pre_noops_without_human_message(self) -> None:
        action = _RecordingAction()
        _apply_llm_pre([AIMessage(content="x")], _fail_eval, action, "g")
        assert action.called is False

    def test_llm_pre_noops_on_empty_text(self) -> None:
        action = _RecordingAction()
        _apply_llm_pre([HumanMessage(content="")], _fail_eval, action, "g")
        assert action.called is False

    def test_llm_pre_swallows_evaluator_exception(self) -> None:
        action = _RecordingAction()
        _apply_llm_pre([HumanMessage(content="hi")], _raise_eval, action, "g")
        assert action.called is False

    def test_llm_pre_applies_modification(self) -> None:
        msg = HumanMessage(content="hi a@b.com")
        _apply_llm_pre([msg], _fail_eval, _ModifyingAction(), "g")
        assert msg.content == "MODIFIED"

    # --- LLM post ---
    def test_llm_post_noops_on_nonstring_content(self) -> None:
        action = _RecordingAction()
        _apply_llm_post(
            AIMessage(content=[{"type": "text", "text": "x"}]),
            _fail_eval,
            action,
            "g",
        )
        assert action.called is False

    def test_llm_post_swallows_evaluator_exception(self) -> None:
        action = _RecordingAction()
        _apply_llm_post(AIMessage(content="ans"), _raise_eval, action, "g")
        assert action.called is False

    def test_llm_post_applies_modification(self) -> None:
        resp = AIMessage(content="ans a@b.com")
        _apply_llm_post(resp, _fail_eval, _ModifyingAction(), "g")
        assert resp.content == "MODIFIED"

    # --- Agent input ---
    def test_agent_input_noops_on_empty_text(self) -> None:
        action = _RecordingAction()
        _apply_agent_input_guardrail(
            {"messages": [HumanMessage(content="")]}, _fail_eval, action, "g"
        )
        assert action.called is False

    def test_agent_input_swallows_evaluator_exception(self) -> None:
        action = _RecordingAction()
        _apply_agent_input_guardrail(
            {"messages": [HumanMessage(content="hi")]}, _raise_eval, action, "g"
        )
        assert action.called is False

    def test_agent_input_applies_modification(self) -> None:
        msg = HumanMessage(content="hi a@b.com")
        _apply_agent_input_guardrail(
            {"messages": [msg]}, _fail_eval, _ModifyingAction(), "g"
        )
        assert msg.content == "MODIFIED"

    # --- Agent output ---
    def test_agent_output_swallows_evaluator_exception(self) -> None:
        action = _RecordingAction()
        _apply_agent_output_guardrail(
            {"messages": [AIMessage(content="ans")]}, _raise_eval, action, "g"
        )
        assert action.called is False

    def test_agent_output_applies_modification(self) -> None:
        msg = AIMessage(content="ans a@b.com")
        _apply_agent_output_guardrail(
            {"messages": [HumanMessage(content="hi"), msg]},
            _fail_eval,
            _ModifyingAction(),
            "g",
        )
        assert msg.content == "MODIFIED"

    # --- multimodal text extraction (list content) ---
    def test_input_payload_from_multimodal_human_message(self) -> None:
        msg = HumanMessage(content=[{"type": "text", "text": "hi there"}])
        assert _input_payload_from_messages([msg]) == json.dumps("hi there")


# ---------------------------------------------------------------------------
# LangChainGuardrailAdapter.wrap dispatch
# ---------------------------------------------------------------------------


class TestAdapterWrapDispatch:
    def test_recognize(self) -> None:
        adapter = LangChainGuardrailAdapter()
        assert adapter.recognize(my_tool) is True
        assert adapter.recognize(object()) is False

    def test_wrap_unknown_target_returns_unchanged(self) -> None:
        adapter = LangChainGuardrailAdapter()
        sentinel = object()
        assert (
            adapter.wrap(
                sentinel,
                _fail_eval,
                _RecordingAction(),
                "g",
                GuardrailExecutionStage.PRE,
            )
            is sentinel
        )

    def test_wrap_dispatches_stategraph(self) -> None:
        from langgraph.graph import StateGraph
        from typing_extensions import TypedDict

        class _S(TypedDict):
            messages: list[Any]

        builder = StateGraph(_S)
        adapter = LangChainGuardrailAdapter()
        result = adapter.wrap(
            builder, _fail_eval, _RecordingAction(), "g", GuardrailExecutionStage.PRE
        )
        assert result is builder
