"""Tests for LLM instrumentor span parenting and get_span_or_root priority."""

from unittest.mock import ANY, MagicMock, patch
from uuid import UUID, uuid4

from opentelemetry.trace import INVALID_SPAN

from uipath_agents._observability.llmops.instrumentors.base import InstrumentationState
from uipath_agents._observability.llmops.instrumentors.llm_instrumentor import (
    LlmSpanInstrumentor,
)
from uipath_agents._observability.llmops.spans import SpanKeys

# --- get_span_or_root priority ---


class TestGetSpanOrRoot:
    """Context stack takes priority over spans dict."""

    def test_context_stack_preferred_over_spans_dict(self) -> None:
        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)

        run_id = uuid4()
        dict_span = MagicMock(name="dict_span")
        ctx_span = MagicMock(name="ctx_span")
        state.spans[run_id] = dict_span
        state.agent_span = MagicMock(name="agent_span")

        with patch(
            "uipath_agents._observability.llmops.callback._get_current_span",
            return_value=ctx_span,
        ):
            result = state.get_span_or_root(run_id)

        assert result is ctx_span

    def test_falls_back_to_spans_dict_when_no_context(self) -> None:
        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)

        run_id = uuid4()
        dict_span = MagicMock(name="dict_span")
        state.spans[run_id] = dict_span
        state.agent_span = MagicMock(name="agent_span")

        with patch(
            "uipath_agents._observability.llmops.callback._get_current_span",
            return_value=None,
        ):
            result = state.get_span_or_root(run_id)

        assert result is dict_span

    def test_falls_back_to_agent_span_when_nothing_matches(self) -> None:
        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)

        agent_span = MagicMock(name="agent_span")
        state.agent_span = agent_span

        with patch(
            "uipath_agents._observability.llmops.callback._get_current_span",
            return_value=None,
        ):
            result = state.get_span_or_root(uuid4())

        assert result is agent_span

    def test_falls_back_to_agent_span_when_run_id_none(self) -> None:
        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)

        agent_span = MagicMock(name="agent_span")
        state.agent_span = agent_span

        with patch(
            "uipath_agents._observability.llmops.callback._get_current_span",
            return_value=None,
        ):
            result = state.get_span_or_root(None)

        assert result is agent_span


# --- Model run parenting ---


def _make_instrumentor(
    agent_span: MagicMock,
    get_span_or_root_return: MagicMock,
) -> tuple[LlmSpanInstrumentor, MagicMock, InstrumentationState]:
    """Create LlmSpanInstrumentor with mocked span factory."""
    mock_span_factory = MagicMock()
    mock_span_factory.start_llm_call.return_value = MagicMock(name="llm_span")
    mock_span_factory.start_model_run.return_value = MagicMock(name="model_span")

    state = InstrumentationState(span_factory=mock_span_factory)
    state.agent_span = agent_span

    instrumentor = LlmSpanInstrumentor(
        state=state,
        close_container=MagicMock(),
    )
    return instrumentor, mock_span_factory, state


class TestModelRunParenting:
    """Model run parents under llm_span for top-level calls, under tool span for inner calls."""

    def test_top_level_call_parents_model_under_llm_span(self) -> None:
        """When parent is agent_span (top-level), model run nests under llm_span."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)
        llm_span = mock_factory.start_llm_call.return_value

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="hello", type="human")]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        model_call = mock_factory.start_model_run.call_args
        assert model_call.kwargs["parent_span"] is llm_span

    def test_inner_call_parents_model_under_tool_span(self) -> None:
        """When parent is a tool span (not agent_span), model run nests under tool span."""
        agent_span = MagicMock(name="agent_span")
        tool_span = MagicMock(name="tool_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, tool_span)

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=tool_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="analyze", type="human")]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        model_call = mock_factory.start_model_run.call_args
        assert model_call.kwargs["parent_span"] is tool_span

    def test_guardrail_path_parents_model_under_llm_span(self) -> None:
        """When reusing guardrail-created llm span, parent stays None → model under llm_span."""
        mock_span_factory = MagicMock()
        guardrail_llm_span = MagicMock(name="guardrail_llm_span")
        mock_span_factory.start_model_run.return_value = MagicMock(name="model_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock(name="agent_span")
        state.current_llm_span = guardrail_llm_span
        state.llm_span_from_guardrail = True

        instrumentor = LlmSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        with patch(
            "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="guarded", type="human")]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        model_call = mock_span_factory.start_model_run.call_args
        assert model_call.kwargs["parent_span"] is guardrail_llm_span


# --- on_llm_end span lifecycle ---

_PATCH_HIERARCHY = "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
_PATCH_USAGE = "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.set_usage_attributes"
_PATCH_TOOL_CALLS = "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.set_tool_calls_attributes"


def _setup_on_llm_end() -> tuple[
    LlmSpanInstrumentor, MagicMock, InstrumentationState, UUID, MagicMock, MagicMock
]:
    """Create instrumentor with pre-populated spans ready for on_llm_end."""
    agent_span = MagicMock(name="agent_span")
    instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)
    run_id = uuid4()
    model_span = MagicMock(name="model_span")
    llm_span = MagicMock(name="llm_span")
    state.spans[run_id] = llm_span
    state.spans[SpanKeys.model(run_id)] = model_span
    return instrumentor, mock_factory, state, run_id, model_span, llm_span


class TestOnLlmEnd:
    """on_llm_end always closes spans, even if attribute-setting throws."""

    def test_happy_path_ends_both_spans_ok(self) -> None:
        instrumentor, mock_factory, state, run_id, model_span, llm_span = (
            _setup_on_llm_end()
        )

        with patch(_PATCH_HIERARCHY), patch(_PATCH_USAGE), patch(_PATCH_TOOL_CALLS):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id)

        mock_factory.end_span_ok.assert_any_call(model_span)
        mock_factory.end_span_ok.assert_any_call(llm_span)
        mock_factory.end_span_error.assert_not_called()
        assert run_id not in state.spans
        assert SpanKeys.model(run_id) not in state.spans

    def test_usage_attributes_throws_still_closes_both_spans(self) -> None:
        instrumentor, mock_factory, state, run_id, model_span, llm_span = (
            _setup_on_llm_end()
        )

        with (
            patch(_PATCH_HIERARCHY),
            patch(_PATCH_USAGE, side_effect=RuntimeError("boom")),
        ):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id)

        mock_factory.end_span_ok.assert_not_called()
        mock_factory.end_span_error.assert_any_call(model_span, ANY)
        mock_factory.end_span_error.assert_any_call(llm_span, ANY)

    def test_end_span_ok_model_throws_llm_span_still_closed(self) -> None:
        instrumentor, mock_factory, state, run_id, model_span, llm_span = (
            _setup_on_llm_end()
        )
        mock_factory.end_span_ok.side_effect = RuntimeError("end failed")

        with patch(_PATCH_HIERARCHY), patch(_PATCH_USAGE), patch(_PATCH_TOOL_CALLS):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id)

        mock_factory.end_span_error.assert_any_call(llm_span, ANY)

    def test_non_recording_span_not_error_closed_on_exception(self) -> None:
        instrumentor, mock_factory, state, run_id, model_span, _ = _setup_on_llm_end()
        state.spans[run_id] = INVALID_SPAN  # replace llm_span with NonRecordingSpan

        with (
            patch(_PATCH_HIERARCHY),
            patch(_PATCH_USAGE, side_effect=RuntimeError("boom")),
        ):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id)

        for call in mock_factory.end_span_error.call_args_list:
            assert call.args[0] is not INVALID_SPAN
