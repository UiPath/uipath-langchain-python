"""LangChain callback for LLMOps span instrumentation.

Injects into LangGraph execution to create LLMOps-schema spans
without modifying the uipath-langchain library.

Key feature: Spans are created on START (not just end) for real-time visibility.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import langchain_core.callbacks
import langchain_core.runnables.config
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span

from .instrumentors import (
    GuardrailSpanInstrumentor,
    InstrumentationState,
    LlmSpanInstrumentor,
    ToolSpanInstrumentor,
)
from .span_hierarchy import SpanHierarchyManager
from .spans import LlmOpsSpanFactory

__all__ = ["LlmOpsInstrumentationCallback", "get_current_run_id"]

logger = logging.getLogger(__name__)


def get_current_run_id() -> Optional[UUID]:
    """Get current run_id from langchain's internal runnable config."""
    config = langchain_core.runnables.config.var_child_runnable_config.get()
    if not isinstance(config, dict):
        return None
    for v in config.values():
        if not isinstance(v, langchain_core.callbacks.BaseCallbackManager):
            continue
        if run_id := v.parent_run_id:
            return run_id
    return None


def _get_current_span() -> Optional[Span]:
    run_id = get_current_run_id()
    if run_id:
        return SpanHierarchyManager.current(run_id)
    return None


def _get_ancestor_spans() -> List[Span]:
    run_id = get_current_run_id()
    if run_id:
        return SpanHierarchyManager.ancestors(run_id)
    return []


class LlmOpsInstrumentationCallback(BaseCallbackHandler):
    """LangChain callback that creates LLMOps-schema OpenTelemetry spans.

    Delegates to specialized instrumentors for LLM, Tool, and Guardrail events.
    Maintains shared state via InstrumentationState that instrumentors access.
    """

    def __init__(
        self,
        span_factory: LlmOpsSpanFactory,
        enriched_properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # Create shared state
        self._state = InstrumentationState(
            span_factory=span_factory,
            enriched_properties=enriched_properties or {},
        )

        # Create instrumentors
        self._guardrail_instrumentor = GuardrailSpanInstrumentor(self._state)
        self._llm_instrumentor = LlmSpanInstrumentor(
            self._state,
            close_container=self._guardrail_instrumentor.close_container,
        )
        self._tool_instrumentor = ToolSpanInstrumentor(
            self._state,
            close_container=self._guardrail_instrumentor.close_container,
        )

    # --- Setup Methods ---

    def set_agent_span(
        self, agent_span: Span, run_id: UUID, prompts_captured: bool = False
    ) -> None:
        """Set the root agent span and reset state for new run."""
        self._state.reset_for_new_run(agent_span, run_id, prompts_captured)
        SpanHierarchyManager.initialize(run_id, agent_span)

    def set_enriched_properties(self, enriched_properties: Dict[str, Any]) -> None:
        self._state.enriched_properties = enriched_properties

    # --- Resume Context Methods ---

    def set_resume_context(
        self,
        tool_name: str,
        trace_id: Optional[str] = None,
        tool_span_data: Optional[Dict[str, Any]] = None,
        process_span_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set resume context - skip creating spans for this tool on first encounter."""
        self._state.resume_tool_name = tool_name
        self._state.resumed_trace_id = trace_id
        self._state.resumed_tool_span_data = tool_span_data
        self._state.resumed_process_span_data = process_span_data

    def set_escalation_resume_context(
        self,
        trace_id: str,
        escalation_span_data: Dict[str, Any],
    ) -> None:
        """Set escalation resume context for guardrail HITL resume."""
        self._state.resumed_escalation_trace_id = trace_id
        self._state.resumed_escalation_span_data = escalation_span_data

    def get_pending_tool_info(
        self,
    ) -> Tuple[Optional[str], Optional[Span], Optional[Span]]:
        return (
            self._state.pending_tool_name,
            self._state.pending_tool_span,
            self._state.pending_process_span,
        )

    def get_pending_escalation_info(
        self,
    ) -> Tuple[Optional[Span], Optional[Dict[str, str]]]:
        return self._state.pending_escalation_span, self._state.pending_escalation_info

    def complete_escalation(
        self,
        review_outcome: str,
        reviewed_by: Optional[str] = None,
        review_reason: Optional[Any] = None,
    ) -> None:
        """Complete the pending escalation span with review results."""
        if self._state.pending_escalation_span:
            self._state.span_factory.end_guardrail_escalation(
                self._state.pending_escalation_span,
                review_outcome=review_outcome,
                reviewed_by=reviewed_by,
                review_reason=review_reason,
            )
            self._state.pending_escalation_span = None
            self._state.pending_escalation_info = None

    def resumed_spans_completed(self) -> bool:
        """Check if resumed tool spans were already completed."""
        return (
            self._state.resumed_trace_id is None
            and self._state.resumed_tool_span_data is None
            and self._state.resumed_process_span_data is None
        )

    # --- Container Management ---

    def cleanup_containers(self) -> None:
        """Close all remaining open guardrail container spans."""
        self._guardrail_instrumentor.cleanup_containers()

    # --- LLM Events ---

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
        self._llm_instrumentor.on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

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
        self._llm_instrumentor.on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._llm_instrumentor.on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._llm_instrumentor.on_llm_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    # --- Tool Events ---

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._tool_instrumentor.on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._tool_instrumentor.on_tool_end(
            output,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._tool_instrumentor.on_tool_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    # --- Chain Events ---

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._guardrail_instrumentor.on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._guardrail_instrumentor.on_chain_end(
            outputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._guardrail_instrumentor.on_chain_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    # --- Cleanup ---

    def cleanup(self) -> None:
        """Clean up all spans and state."""
        # End orphaned spans
        for span in self._state.spans.values():
            if (
                span is self._state.pending_tool_span
                or span is self._state.pending_process_span
            ):
                continue
            try:
                span.end()
            except Exception as e:
                logger.debug("Failed to end span during cleanup: %s", e)
        self._state.spans.clear()

        # Clear pending references
        self._state.pending_tool_span = None
        self._state.pending_process_span = None
        self._state.pending_tool_name = None

        # Clear resume-related fields
        self._state.resumed_trace_id = None
        self._state.resumed_tool_span_data = None
        self._state.resumed_process_span_data = None

        # End guardrail spans
        for span in self._state.guardrail_spans.values():
            try:
                span.end()
            except Exception as e:
                logger.debug("Failed to end guardrail span during cleanup: %s", e)
        self._state.guardrail_spans.clear()
        self._state.guardrail_info.clear()
        self._state.guardrail_metadata.clear()

        # End pending guardrail action spans
        for span, _ in self._state.pending_guardrail_actions.values():
            try:
                span.end()
            except Exception as e:
                logger.debug(
                    "Failed to end guardrail action span during cleanup: %s", e
                )
        self._state.pending_guardrail_actions.clear()

        # End guardrail containers
        for span in self._state.guardrail_containers.values():
            try:
                span.end()
            except Exception as e:
                logger.debug("Failed to end guardrail container during cleanup: %s", e)
        self._state.guardrail_containers.clear()

        # End escalate action spans
        for span in self._state.escalate_action_run_ids.values():
            try:
                span.end()
            except Exception as e:
                logger.debug("Failed to end escalate action span during cleanup: %s", e)
        self._state.escalate_action_run_ids.clear()

        # End orphaned placeholder LLM span
        if self._state.llm_span_from_guardrail and self._state.current_llm_span:
            try:
                self._state.current_llm_span.end()
            except Exception as e:
                logger.debug("Failed to end placeholder LLM span during cleanup: %s", e)
            self._state.llm_span_from_guardrail = False

        # End orphaned placeholder tool span
        if self._state.tool_span_from_guardrail and self._state.current_tool_span:
            try:
                self._state.current_tool_span.end()
            except Exception as e:
                logger.debug(
                    "Failed to end placeholder tool span during cleanup: %s", e
                )
            self._state.tool_span_from_guardrail = False

        self._state.current_llm_span = None
        self._state.current_tool_span = None

        # Clean up span stack
        if self._state.agent_run_id:
            SpanHierarchyManager.cleanup(self._state.agent_run_id)
