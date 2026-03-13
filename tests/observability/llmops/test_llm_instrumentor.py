"""Tests for LLM instrumentor span parenting and get_span_or_root priority."""

import json
from typing import Any
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
    mock_span_factory.start_model_run.return_value = (
        MagicMock(name="model_span"),
        None,
    )

    state = InstrumentationState(span_factory=mock_span_factory)
    state.agent_span = agent_span

    instrumentor = LlmSpanInstrumentor(
        state=state,
        close_container=MagicMock(),
    )
    return instrumentor, mock_span_factory, state


class TestModelRunParenting:
    """Model run always parents under llm_span, regardless of call depth."""

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

    def test_inner_call_parents_model_under_llm_span(self) -> None:
        """When parent is a tool span (inner call), model run still nests under llm_span."""
        agent_span = MagicMock(name="agent_span")
        tool_span = MagicMock(name="tool_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, tool_span)
        llm_span = mock_factory.start_llm_call.return_value

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
        assert model_call.kwargs["parent_span"] is llm_span

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


# --- Base64 sanitization ---


class TestBase64Sanitization:
    """on_chat_model_start sanitizes base64 file data from multimodal content."""

    def test_multimodal_content_base64_sanitized(self) -> None:
        """Base64 image data in multimodal message parts is replaced with placeholder."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        base64_payload = "A" * 2000  # long base64-like string
        multimodal_content = [
            {"type": "text", "text": "Analyze this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_payload}"},
            },
        ]
        msg = MagicMock(content=multimodal_content, type="human")

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        input_value = mock_factory.start_llm_call.call_args.kwargs["input"]
        assert base64_payload not in input_value
        assert "<base64 data omitted>" in input_value

    def test_data_uri_in_dict_content_sanitized(self) -> None:
        """data: URI with base64 in a dict-based content part is sanitized."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        content = [
            {"type": "text", "text": "describe the file"},
            {
                "type": "image_url",
                "data": "data:application/pdf;base64,JVBERi0xLjQK" + "A" * 1500,
            },
        ]
        msg = MagicMock(content=content, type="human")

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        input_value = mock_factory.start_llm_call.call_args.kwargs["input"]
        assert "JVBERi0" not in input_value

    def test_raw_base64_in_data_key_sanitized(self) -> None:
        """Raw base64 bytes in a 'data' key are replaced with placeholder."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        raw_base64 = "A" * 2000
        content = [
            {"type": "text", "text": "describe the file"},
            {"type": "file", "data": raw_base64},
        ]
        msg = MagicMock(content=content, type="human")

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        input_value = mock_factory.start_llm_call.call_args.kwargs["input"]
        assert raw_base64 not in input_value
        assert "<base64 data omitted>" in input_value

    def test_multimodal_content_serialized_as_json(self) -> None:
        """Multimodal list content is serialized as JSON, not Python repr."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        multimodal_content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = MagicMock(content=multimodal_content, type="human")

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        input_value = mock_factory.start_llm_call.call_args.kwargs["input"]
        # Must be valid JSON, not Python repr (which uses single quotes)
        parsed = json.loads(input_value)
        assert isinstance(parsed, list)
        assert parsed[0]["type"] == "text"
        assert parsed[0]["text"] == "What is in this image?"

    def test_non_serializable_content_falls_back_to_str(self) -> None:
        """Non-JSON-serializable content falls back to str() without dropping spans."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        # Content with a custom object that serialize_json cannot handle
        custom_obj = object()
        multimodal_content = [
            {"type": "text", "text": "hello"},
            {"type": "custom", "value": custom_obj},
        ]
        msg = MagicMock(content=multimodal_content, type="human")

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        # Span must still be created (not dropped)
        mock_factory.start_llm_call.assert_called_once()
        input_value = mock_factory.start_llm_call.call_args.kwargs["input"]
        assert "hello" in input_value

    def test_plain_text_content_not_affected(self) -> None:
        """Plain text content passes through without modification."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        msg = MagicMock(content="What is 2+2?", type="human")

        run_id = uuid4()
        with (
            patch(
                "uipath_agents._observability.llmops.instrumentors.llm_instrumentor.SpanHierarchyManager"
            ),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        input_value = mock_factory.start_llm_call.call_args.kwargs["input"]
        assert input_value == "What is 2+2?"


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

        for c in mock_factory.end_span_error.call_args_list:
            assert c.args[0] is not INVALID_SPAN

    def test_current_llm_span_preserved_after_on_llm_end(self) -> None:
        """current_llm_span stays set after on_llm_end — POST guardrails need it;
        next on_chat_model_start overwrites it."""
        instrumentor, mock_factory, state, run_id, model_span, llm_span = (
            _setup_on_llm_end()
        )
        state.current_llm_span = llm_span

        with patch(_PATCH_HIERARCHY), patch(_PATCH_USAGE), patch(_PATCH_TOOL_CALLS):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id)

        assert state.current_llm_span is llm_span

    def test_current_llm_span_preserved_after_on_llm_error(self) -> None:
        """current_llm_span stays set after on_llm_error — POST guardrails need it;
        next on_chat_model_start overwrites it."""
        instrumentor, mock_factory, state, run_id, model_span, llm_span = (
            _setup_on_llm_end()
        )
        state.current_llm_span = llm_span

        with patch(_PATCH_HIERARCHY):
            instrumentor.on_llm_error(
                RuntimeError("test"), run_id=run_id, parent_run_id=None
            )

        assert state.current_llm_span is llm_span


# --- Multi-turn LLM span isolation ---


class TestMultiTurnLlmSpanIsolation:
    """Simulates LLM → tool → LLM flow to verify second call gets a fresh span."""

    def test_second_llm_call_creates_fresh_span_after_first_completes(self) -> None:
        """After on_llm_end clears current_llm_span, the next on_chat_model_start
        must create a new llm_span (not reuse stale state)."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        first_llm_span = MagicMock(name="first_llm_span")
        second_llm_span = MagicMock(name="second_llm_span")
        mock_factory.start_llm_call.side_effect = [first_llm_span, second_llm_span]

        run_id_1 = uuid4()
        parent_run_id = uuid4()

        # --- First LLM call ---
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="first call", type="human")]],
                run_id=run_id_1,
                parent_run_id=parent_run_id,
            )
        assert state.current_llm_span is first_llm_span

        # Complete first LLM call
        state.spans[SpanKeys.model(run_id_1)] = MagicMock(name="model_span_1")
        with patch(_PATCH_HIERARCHY), patch(_PATCH_USAGE), patch(_PATCH_TOOL_CALLS):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id_1)
        assert state.current_llm_span is first_llm_span

        # --- Second LLM call (after tool execution) ---
        run_id_2 = uuid4()
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="second call", type="human")]],
                run_id=run_id_2,
                parent_run_id=parent_run_id,
            )

        # Second call must get a NEW span, not the first one
        assert state.current_llm_span is second_llm_span
        assert state.current_llm_span is not first_llm_span
        assert mock_factory.start_llm_call.call_count == 2

    def test_second_llm_call_after_error_creates_fresh_span(self) -> None:
        """After on_llm_error clears current_llm_span, next call gets a fresh span."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        first_llm_span = MagicMock(name="first_llm_span")
        second_llm_span = MagicMock(name="second_llm_span")
        mock_factory.start_llm_call.side_effect = [first_llm_span, second_llm_span]

        run_id_1 = uuid4()
        parent_run_id = uuid4()

        # --- First LLM call ---
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="will fail", type="human")]],
                run_id=run_id_1,
                parent_run_id=parent_run_id,
            )

        # Error out first LLM call
        state.spans[SpanKeys.model(run_id_1)] = MagicMock(name="model_span_1")
        with patch(_PATCH_HIERARCHY):
            instrumentor.on_llm_error(
                RuntimeError("LLM timeout"), run_id=run_id_1, parent_run_id=None
            )
        assert state.current_llm_span is first_llm_span

        # --- Retry: second LLM call ---
        run_id_2 = uuid4()
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="retry", type="human")]],
                run_id=run_id_2,
                parent_run_id=parent_run_id,
            )

        assert state.current_llm_span is second_llm_span
        assert mock_factory.start_llm_call.call_count == 2

    def test_model_spans_correctly_nested_across_multi_turn(self) -> None:
        """Each LLM iteration's model_run span nests under its own llm_span."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        first_llm = MagicMock(name="first_llm")
        second_llm = MagicMock(name="second_llm")
        first_model = MagicMock(name="first_model")
        second_model = MagicMock(name="second_model")

        mock_factory.start_llm_call.side_effect = [first_llm, second_llm]
        mock_factory.start_model_run.side_effect = [
            (first_model, MagicMock()),
            (second_model, MagicMock()),
        ]

        parent_run_id = uuid4()

        # First turn
        run_id_1 = uuid4()
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="turn 1", type="human")]],
                run_id=run_id_1,
                parent_run_id=parent_run_id,
            )

        first_model_call = mock_factory.start_model_run.call_args_list[0]
        assert first_model_call.kwargs["parent_span"] is first_llm

        # Complete first turn
        state.spans[SpanKeys.model(run_id_1)] = first_model
        with patch(_PATCH_HIERARCHY), patch(_PATCH_USAGE), patch(_PATCH_TOOL_CALLS):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id_1)

        # Second turn
        run_id_2 = uuid4()
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[MagicMock(content="turn 2", type="human")]],
                run_id=run_id_2,
                parent_run_id=parent_run_id,
            )

        second_model_call = mock_factory.start_model_run.call_args_list[1]
        assert second_model_call.kwargs["parent_span"] is second_llm


# --- on_chat_model_start sanitization ---


class TestOnChatModelStartSanitization:
    """on_chat_model_start sanitizes message content before storing as span input."""

    def _invoke_chat_model_start(
        self,
        content: Any,
        msg_type: str = "human",
    ) -> MagicMock:
        """Helper: call on_chat_model_start and return the mock span factory."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, mock_factory, state = _make_instrumentor(agent_span, agent_span)

        msg = MagicMock(content=content, type=msg_type)
        run_id = uuid4()
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )
        return mock_factory

    def test_plain_string_content_passed_as_input(self) -> None:
        """Test that plain string content is passed unchanged to the LLM span."""
        mock_factory = self._invoke_chat_model_start("What is 2+2?")
        input_text = mock_factory.start_llm_call.call_args.kwargs["input"]
        assert input_text == "What is 2+2?"

    def test_data_uri_in_string_content_is_sanitized(self) -> None:
        """Test that a data URI string is replaced with placeholder."""
        mock_factory = self._invoke_chat_model_start(
            "data:image/png;base64," + "A" * 5000
        )
        input_text = mock_factory.start_llm_call.call_args.kwargs["input"]
        assert input_text == "<base64 data omitted>"

    def test_multimodal_list_content_is_sanitized(self) -> None:
        """Test that multimodal list content with base64 image is sanitized."""
        content = [
            {"type": "text", "text": "Describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "B" * 5000},
            },
        ]
        mock_factory = self._invoke_chat_model_start(content)
        input_text = mock_factory.start_llm_call.call_args.kwargs["input"]
        # List content gets str() after sanitization
        assert "<base64 data omitted>" in input_text
        assert "Describe this" in input_text
        # Original multi-MB data should not be present
        assert "B" * 5000 not in input_text

    def test_user_prompt_attribute_is_sanitized(self) -> None:
        """Test that userPrompt on agent span is also sanitized."""
        agent_span = MagicMock(name="agent_span")
        instrumentor, _, state = _make_instrumentor(agent_span, agent_span)

        content = [
            {"type": "text", "text": "Analyze this file"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "C" * 3000},
            },
        ]
        msg = MagicMock(content=content, type="human")
        run_id = uuid4()
        with (
            patch(_PATCH_HIERARCHY),
            patch.object(state, "get_span_or_root", return_value=agent_span),
        ):
            instrumentor.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}},
                messages=[[msg]],
                run_id=run_id,
                parent_run_id=uuid4(),
            )

        # Find the userPrompt set_attribute call on agent_span
        user_prompt_calls = [
            c
            for c in agent_span.set_attribute.call_args_list
            if c.args[0] == "userPrompt"
        ]
        assert len(user_prompt_calls) == 1
        prompt_value = user_prompt_calls[0].args[1]
        assert "C" * 3000 not in prompt_value
        assert "<base64 data omitted>" in prompt_value


# --- POST guardrail LLM span preservation ---


class TestPostGuardrailLlmSpanPreservation:
    """Regression: current_llm_span must survive on_llm_end so POST guardrails can parent under it."""

    def test_current_llm_span_preserved_after_on_llm_end_for_post_guardrails(
        self,
    ) -> None:
        """After on_llm_end, current_llm_span is still set so POST guardrails parent under it."""
        instrumentor, mock_factory, state, run_id, model_span, llm_span = (
            _setup_on_llm_end()
        )
        state.current_llm_span = llm_span

        with patch(_PATCH_HIERARCHY), patch(_PATCH_USAGE), patch(_PATCH_TOOL_CALLS):
            instrumentor.on_llm_end(MagicMock(), run_id=run_id)

        # POST guardrails read current_llm_span as parent — must not be None
        assert state.current_llm_span is llm_span
        assert state.current_llm_span is not None

    def test_close_container_clears_llm_span_when_post_container_closed(
        self,
    ) -> None:
        """When close_container finds an actual POST container, it clears current_llm_span."""
        from uipath.core.guardrails import GuardrailScope
        from uipath_langchain.agent.guardrails.types import ExecutionStage

        from uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor import (
            GuardrailSpanInstrumentor,
        )

        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock(name="agent_span")

        llm_span = MagicMock(name="llm_span")
        state.current_llm_span = llm_span

        # Simulate a POST container existing in state
        container_span = MagicMock(name="post_container")
        state.guardrail_containers[
            (GuardrailScope.LLM, ExecutionStage.POST_EXECUTION)
        ] = container_span

        guardrail_instrumentor = GuardrailSpanInstrumentor(state=state)
        guardrail_instrumentor.close_container(
            GuardrailScope.LLM, ExecutionStage.POST_EXECUTION
        )

        # Container closed → current_llm_span cleared
        assert state.current_llm_span is None
        mock_span_factory.end_span_ok.assert_called_once_with(container_span)
