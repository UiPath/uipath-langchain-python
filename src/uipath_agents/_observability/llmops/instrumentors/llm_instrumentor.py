"""LLM span instrumentor for LLMOps instrumentation."""

import logging
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import NonRecordingSpan
from uipath.core.guardrails import GuardrailScope
from uipath.core.serialization import serialize_json
from uipath_langchain.agent.guardrails.types import ExecutionStage

from ..span_hierarchy import SpanHierarchyManager
from ..spans import SpanKeys
from .attribute_helpers import (
    extract_model_name,
    extract_settings,
    sanitize_file_data,
    set_tool_calls_attributes,
    set_usage_attributes,
)
from .base import BaseSpanInstrumentor, InstrumentationState

logger = logging.getLogger(__name__)


class LlmSpanInstrumentor(BaseSpanInstrumentor):
    """Instruments LLM events with spans: on_llm_start, on_chat_model_start, on_llm_end, on_llm_error.

    Creates nested LLM call + model run spans. The LLM call span is the outer container,
    and the model run span tracks the actual API call inside.

    Span hierarchy:
        AgentRun
        └── LlmCall (outer)
            └── ModelRun (inner - actual API call)
    """

    def __init__(
        self,
        state: InstrumentationState,
        close_container: Callable[[str, str], None],
    ) -> None:
        """Initialize LLM span instrumentor.

        Args:
            state: Shared instrumentation state
            close_container: Callback to close guardrail containers (scope, stage)
        """
        super().__init__(state)
        self._close_container = close_container

    def _model_span_key(self, run_id: UUID) -> UUID:
        """Derive unique key for model span from LLM run_id."""
        return SpanKeys.model(run_id)

    def _start_llm_and_model_spans(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        serialized: Dict[str, Any],
        input_text: Optional[str] = None,
    ) -> None:
        """Create nested LLM call + model run spans.

        If LLM span was created early by llm_pre guardrails, reuses it.
        """
        # Close agent_pre container before LLM starts
        self._close_container(GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION)

        model_name = extract_model_name(serialized)
        max_tokens, temperature = extract_settings(serialized)

        # Check if llmCall was created early by guardrails
        # Only reuse if the flag indicates it was created by guardrails
        parent = None  # Resolved parent; set in non-guardrail path
        if self._state.current_llm_span and self._state.llm_span_from_guardrail:
            llm_span = self._state.current_llm_span
            if input_text:
                llm_span.set_attribute("input", input_text)
            # Clear the flag after reuse - span was consumed by on_llm_start
            self._state.llm_span_from_guardrail = False
            # Keep current_llm_span set for llm_post guardrails to use
        else:
            parent = self._state.get_span_or_root(parent_run_id)
            llm_span = self._span_factory.start_llm_call(
                input=input_text,
                parent_span=parent,
            )
            # Set current_llm_span for HITL to access if needed
            self._state.current_llm_span = llm_span

        self._spans[run_id] = llm_span

        # Close llm_pre container before model span
        self._close_container(GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION)

        # For inner calls (nested inside a tool), parent the model run directly
        # under the tool span. The LLMOps server may drop the intermediate
        # "LLM call" span, so this ensures the "Model run" always has a valid
        # parent in the server trace.
        is_inner_call = parent is not None and parent is not self._agent_span
        model_span = self._span_factory.start_model_run(
            model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            parent_span=parent if is_inner_call else llm_span,
        )
        self._spans[self._model_span_key(run_id)] = model_span
        SpanHierarchyManager.push(run_id, model_span)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start event (non-chat models)."""
        try:
            input_text = prompts[0] if prompts else None
            self._start_llm_and_model_spans(
                run_id, parent_run_id, serialized, input_text=input_text
            )
        except Exception:
            logger.exception("Error in on_llm_start callback")

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chat model start event. Also captures interpolated prompts."""
        try:
            # Extract last message as input
            input_text = None
            if messages and messages[0]:
                last_msg = messages[0][-1]
                content = last_msg.content
                input_text = content if isinstance(content, str) else str(content)

            self._start_llm_and_model_spans(
                run_id, parent_run_id, serialized, input_text=input_text
            )

            if self._agent_span and not self._state.prompts_captured:
                self._capture_interpolated_prompts(messages)
        except Exception:
            logger.exception("Error in on_chat_model_start callback")

    def _capture_interpolated_prompts(self, messages: List[List[BaseMessage]]) -> None:
        """Extract and set interpolated prompts on the AgentRun span."""
        if not messages or not messages[0]:
            return

        for msg in messages[0]:
            if msg.type == "human" and self._agent_span:
                sanitized = sanitize_file_data(msg.content)
                content = serialize_json(sanitized)
                self._agent_span.set_attribute("userPrompt", content)

        self._state.prompts_captured = True

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM end event."""
        try:
            SpanHierarchyManager.pop(run_id)

            # Close inner span first, then outer
            model_span = self._spans.pop(self._model_span_key(run_id), None)
            if model_span:
                set_usage_attributes(model_span, response)
                set_tool_calls_attributes(model_span, response)
                self._span_factory.end_span_ok(model_span)

            llm_span = self._spans.pop(run_id, None)
            if llm_span:
                # Handle NonRecordingSpan (resumed LLM span) - use upsert_span_complete_by_data
                if isinstance(llm_span, NonRecordingSpan):
                    llm_span_data = self._state.resumed_llm_span_data
                    trace_id = (
                        self._state.resumed_escalation_trace_id
                        or self._state.resumed_trace_id
                    )
                    if llm_span_data and trace_id:
                        self._span_factory.upsert_span_complete_by_data(
                            trace_id=trace_id,
                            span_data=llm_span_data,
                        )
                else:
                    self._span_factory.end_span_ok(llm_span)

            self._close_container(GuardrailScope.LLM, ExecutionStage.POST_EXECUTION)

        except Exception:
            logger.exception("Error in on_llm_end callback")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM error event."""
        try:
            SpanHierarchyManager.pop(run_id)

            exc = error if isinstance(error, Exception) else Exception(str(error))

            model_span = self._spans.pop(self._model_span_key(run_id), None)
            if model_span:
                self._span_factory.end_span_error(model_span, exc)

            llm_span = self._spans.pop(run_id, None)
            if llm_span:
                self._span_factory.end_span_error(llm_span, exc)

            self._close_container(GuardrailScope.LLM, ExecutionStage.POST_EXECUTION)

        except Exception:
            logger.exception("Error in on_llm_error callback")
