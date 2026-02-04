"""Tool span instrumentor for LLMOps instrumentation."""

import json
import logging
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from uipath.core.guardrails import GuardrailScope
from uipath.eval.mocks.mockable import MOCKED_ANNOTATION_KEY
from uipath_langchain.agent.guardrails.types import ExecutionStage

from ..span_hierarchy import SpanHierarchyManager
from ..spans import SpanKeys
from .attribute_helpers import (
    get_tool_type_value,
    parse_tool_arguments,
    set_escalation_task_info,
    set_process_job_info,
    set_tool_result,
)
from .base import BaseSpanInstrumentor, InstrumentationState

logger = logging.getLogger(__name__)


class ToolSpanInstrumentor(BaseSpanInstrumentor):
    """Instruments tool events with spans: on_tool_start, on_tool_end, on_tool_error.

    Creates tool call spans with optional child spans for escalation, process,
    agent, or integration tools.

    Span hierarchy:
        AgentRun
        └── ToolCall (outer)
            └── [Escalation|Process|Agent|Integration] (child)
    """

    def __init__(
        self,
        state: InstrumentationState,
        close_container: Callable[[str, str], None],
    ) -> None:
        """Initialize Tool span instrumentor.

        Args:
            state: Shared instrumentation state
            close_container: Callback to close guardrail containers (scope, stage)
        """
        super().__init__(state)
        self._close_container = close_container

    def _interruptible_span_key(self, run_id: UUID) -> UUID:
        """Derive unique key for tool child span."""
        return SpanKeys.tool_child(run_id)

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
        """Handle tool start event."""
        try:
            tool_name = serialized.get("name", "unknown")
            tool_type = metadata.get("tool_type") if metadata else None
            tool_display_name = metadata.get("display_name") if metadata else None

            # Resume mode: skip duplicate span for re-invoked tool
            if (
                self._state.resume_tool_name
                and tool_name == self._state.resume_tool_name
            ):
                logger.debug("Resume mode: skipping span creation for %s", tool_name)
                self._state.resume_tool_name = None
                self._state.reinvoked_tool_run_ids.add(run_id)
                return

            call_id = kwargs.get("tool_call_id")
            arguments = parse_tool_arguments(input_str)
            tool_type_value = get_tool_type_value(tool_type)
            args_schema = metadata.get("args_schema") if metadata else None

            # Check if tool span was created early by tool_pre guardrails
            # Only reuse if the flag indicates it was created by guardrails
            if self._state.current_tool_span and self._state.tool_span_from_guardrail:
                span = self._state.current_tool_span
                span.set_attribute("toolName", tool_name)
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("toolType", tool_type_value)
                if call_id:
                    span.set_attribute("callId", call_id)
                if arguments:
                    span.set_attribute("arguments", json.dumps(arguments))
                self._spans[run_id] = span
                # Clear the flag after reuse - span was consumed by on_tool_start
                self._state.tool_span_from_guardrail = False
                self._close_container(GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION)
            else:
                parent = self._state.get_span_or_root(parent_run_id)
                span = self._span_factory.start_tool_call(
                    tool_name,
                    tool_type_value=tool_type_value,
                    arguments=arguments,
                    call_id=call_id,
                    parent_span=parent,
                    args_schema=args_schema,
                )
                span.set_attribute("tool.name", tool_name)
                self._spans[run_id] = span
                # Set current_tool_span for HITL to access if needed
                self._state.current_tool_span = span

            SpanHierarchyManager.push(run_id, span)

            # Create child span for typed tools
            if tool_type:
                child_span = None
                if tool_type == "escalation" and tool_display_name:
                    child_span = self._span_factory.start_escalation_tool(
                        app_name=tool_display_name,
                        parent_span=span,
                    )
                    self._state.escalation_run_ids.add(run_id)
                elif tool_type == "agent" and tool_display_name:
                    child_span = self._span_factory.start_agent_tool(
                        agent_name=tool_display_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                    self._state.agent_run_ids.add(run_id)
                elif tool_type == "process" and tool_display_name:
                    child_span = self._span_factory.start_process_tool(
                        process_name=tool_display_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                    self._state.process_run_ids.add(run_id)
                elif tool_type == "integration":
                    child_span = self._span_factory.start_integration_tool(
                        tool_name=tool_display_name or tool_name,
                        parent_span=span,
                    )

                if child_span:
                    self._spans[self._interruptible_span_key(run_id)] = child_span
                    SpanHierarchyManager.push(run_id, child_span)
                    # Track as pending for suspend scenario
                    if tool_type in ("escalation", "process", "agent"):
                        self._state.pending_tool_name = tool_name
                        self._state.pending_tool_span = span
                        self._state.pending_process_span = child_span

        except Exception:
            logger.exception("Error in on_tool_start callback")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool end event."""
        try:
            # Handle resumed tool completion
            if run_id in self._state.reinvoked_tool_run_ids:
                self._state.reinvoked_tool_run_ids.discard(run_id)
                self._upsert_resumed_spans_on_completion(output)
                return

            # Close child span first (inner), then tool span (outer)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                if hasattr(child_span, "attributes"):
                    if child_span.attributes.get(MOCKED_ANNOTATION_KEY) and hasattr(
                        child_span, "name"
                    ):
                        child_span.update_name(f"Simulated result: {child_span.name}")
                SpanHierarchyManager.pop(run_id)
                set_tool_result(child_span, output)
                if run_id in self._state.escalation_run_ids:
                    set_escalation_task_info(child_span, output)
                    self._state.escalation_run_ids.discard(run_id)
                if run_id in self._state.process_run_ids:
                    set_process_job_info(child_span, output)
                    self._state.process_run_ids.discard(run_id)
                if run_id in self._state.agent_run_ids:
                    set_process_job_info(child_span, output)
                    self._state.agent_run_ids.discard(run_id)
                self._span_factory.end_span_ok(child_span)
                if child_span == self._state.pending_process_span:
                    self._state.pending_process_span = None

            span = self._spans.pop(run_id, None)
            if span:
                SpanHierarchyManager.pop(run_id)
                set_tool_result(span, output)
                self._span_factory.end_span_ok(span)
                if span == self._state.pending_tool_span:
                    self._state.pending_tool_span = None
                    self._state.pending_tool_name = None

        except Exception:
            logger.exception("Error in on_tool_end callback")

    def _upsert_resumed_spans_on_completion(self, output: Any) -> None:
        """Upsert resumed tool/process spans when tool completes."""
        if not self._state.resumed_trace_id:
            return

        # Upsert process span first (inner)
        if self._state.resumed_process_span_data:
            if output is not None:
                result = (
                    json.dumps(output)
                    if isinstance(output, (dict, list))
                    else str(output)
                )
                self._state.resumed_process_span_data["attributes"]["result"] = result

            self._span_factory.upsert_span_complete_by_data(
                trace_id=self._state.resumed_trace_id,
                span_data=self._state.resumed_process_span_data,
            )
            logger.debug(
                "Upserted resumed process span %s on tool completion",
                self._state.resumed_process_span_data.get("name", "unknown"),
            )

        # Upsert tool span (outer)
        if self._state.resumed_tool_span_data:
            if output is not None:
                result = (
                    json.dumps(output)
                    if isinstance(output, (dict, list))
                    else str(output)
                )
                self._state.resumed_tool_span_data["attributes"]["result"] = result

            self._span_factory.upsert_span_complete_by_data(
                trace_id=self._state.resumed_trace_id,
                span_data=self._state.resumed_tool_span_data,
            )
            logger.debug(
                "Upserted resumed tool span %s on tool completion",
                self._state.resumed_tool_span_data.get("name", "unknown"),
            )

        self._state.resumed_process_span_data = None

    def _is_graph_interrupt(self, error: BaseException) -> bool:
        """Check if the error is a GraphInterrupt (suspend signal)."""
        error_str = str(error)
        error_type = type(error).__name__
        return error_type == "GraphInterrupt" or error_str.startswith("GraphInterrupt(")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool error event."""
        try:
            if run_id in self._state.reinvoked_tool_run_ids:
                self._state.reinvoked_tool_run_ids.discard(run_id)
                return

            # GraphInterrupt = suspend signal, spans kept open
            if self._is_graph_interrupt(error):
                logger.debug(
                    "GraphInterrupt detected for tool, spans kept open for upsert"
                )
                return

            exc = error if isinstance(error, Exception) else Exception(str(error))

            # Close child span first (inner), then tool span (outer)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                SpanHierarchyManager.pop(run_id)
                self._span_factory.end_span_error(child_span, exc)

            span = self._spans.pop(run_id, None)
            if span:
                SpanHierarchyManager.pop(run_id)
                self._span_factory.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_tool_error callback")
