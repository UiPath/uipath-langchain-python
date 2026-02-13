"""Tests for LLM instrumentor span parenting and get_span_or_root priority."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from uipath_agents._observability.llmops.instrumentors.base import InstrumentationState
from uipath_agents._observability.llmops.instrumentors.llm_instrumentor import (
    LlmSpanInstrumentor,
)

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
