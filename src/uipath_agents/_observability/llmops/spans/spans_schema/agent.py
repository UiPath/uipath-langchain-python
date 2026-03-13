"""Agent span schemas.

Handles agent run and agent output spans.
"""

import json
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
)
from uipath.tracing import AttachmentDirection, SpanStatus

from ...instrumentors.attribute_helpers import get_span_attachments
from ..span_attributes import (
    AgentOutputSpanAttributes,
    AgentRunSpanAttributes,
)
from ..span_name import SpanName
from .base import (
    apply_attributes,
    format_span_error,
    reference_id_context,
    uipath_source_context,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AgentSpanSchema",
]


class AgentSpanSchema:
    """Schema for agent-related spans."""

    def __init__(
        self,
        tracer: Tracer,
        upsert_started_fn: Optional[Callable[[Span], bool]] = None,
        upsert_complete_fn: Optional[Callable[[Span, int], bool]] = None,
    ):
        """Initialize agent span schema.

        Args:
            tracer: The OpenTelemetry tracer to use
            upsert_started_fn: Optional function to upsert span on start
            upsert_complete_fn: Optional function to upsert span on complete
        """
        self._tracer = tracer
        self._upsert_started = upsert_started_fn
        self._upsert_complete = upsert_complete_fn

    def _fire_and_forget_upsert(self, span: Span) -> threading.Thread:
        def _upsert() -> None:
            try:
                if self._upsert_started:
                    self._upsert_started(span)
            except Exception:
                logger.debug("Background upsert_started failed", exc_info=True)

        thread = threading.Thread(target=_upsert, daemon=True)
        thread.start()
        return thread

    @contextmanager
    def start_agent_run(
        self,
        agent_name: str,
        *,
        agent_id: Optional[str] = None,
        is_conversational: bool = False,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
    ) -> Generator[Span, None, None]:
        """Start an agent run span (root span for agent execution).

        All LLM calls, tool calls, and guardrails are children of this span.

        Args:
            agent_name: Name of the agent
            agent_id: Unique identifier for the agent instance
            is_conversational: Whether this is a conversational agent
            system_prompt: The system prompt used for the agent
            user_prompt: The user prompt/input for this run
            input_data: Input arguments passed to the agent
            input_schema: JSON schema for agent input
            output_schema: JSON schema for agent output
            source: Execution source (runtime, playground, eval, unknown)

        Yields:
            The OpenTelemetry Span object
        """
        span_name = SpanName.agent_run(agent_name, is_conversational)

        reference_id = agent_id
        token = reference_id_context.set(reference_id)
        source_token = uipath_source_context.set(1)
        attachments = get_span_attachments(
            input_data, input_schema, direction=AttachmentDirection.IN
        )

        # Create typed attributes
        attrs = AgentRunSpanAttributes(
            agent_name=agent_name,
            agent_id=agent_id,
            is_conversational=is_conversational,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input=input_data,
            input_schema=input_schema,
            output_schema=output_schema,
            reference_id=reference_id,
            source=source,
            attachments=attachments,
        )

        agent_span: Optional[Span] = None
        upsert_thread: Optional[threading.Thread] = None
        final_status: int = SpanStatus.OK
        # Use start_span + use_span(end_on_exit=False) so the caller
        # controls when span.end() fires. This prevents
        # LiveTrackingSpanProcessor.on_end() from sending a premature
        # OK upsert when the agent is suspended.
        span = self._tracer.start_span(span_name, kind=SpanKind.INTERNAL)
        try:
            with trace.use_span(span, end_on_exit=False, set_status_on_exception=False):
                agent_span = span
                apply_attributes(span, attrs)
                if self._upsert_started:
                    upsert_thread = self._fire_and_forget_upsert(span)

                try:
                    yield span
                except Exception as e:
                    span.set_attribute("error", format_span_error(e))
                    span.set_status(Status(StatusCode.ERROR, format_span_error(e)))
                    final_status = SpanStatus.ERROR
                    span.end()
                    raise
        finally:
            reference_id_context.reset(token)
            uipath_source_context.reset(source_token)
            if upsert_thread is not None:
                upsert_thread.join()
            # Only upsert if the caller ended the span. Suspended spans
            # are intentionally left open (is_recording=True).
            if (
                self._upsert_complete
                and agent_span is not None
                and not agent_span.is_recording()
            ):
                self._upsert_complete(agent_span, final_status)

    def emit_agent_output(
        self, output: Any, output_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit agent output span (short-lived, captures final output).

        Args:
            output: The agent's output (will be JSON serialized if dict/list)
        """
        # Serialize output to string
        if isinstance(output, (dict, list)):
            output_str = json.dumps(output)
        else:
            output_str = str(output) if output is not None else ""

        with self._tracer.start_as_current_span(
            SpanName.AGENT_OUTPUT,
            kind=SpanKind.INTERNAL,
        ) as span:
            attachments = get_span_attachments(
                output, output_schema, direction=AttachmentDirection.OUT
            )
            attrs = AgentOutputSpanAttributes(
                output=output_str, attachments=attachments
            )
            apply_attributes(span, attrs)
            span.set_status(Status(StatusCode.OK))
