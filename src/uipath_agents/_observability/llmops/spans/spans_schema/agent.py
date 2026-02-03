"""Agent span schemas.

Handles agent run and agent output spans.
"""

import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
)

from ..span_attributes import (
    AgentOutputSpanAttributes,
    AgentRunSpanAttributes,
)
from ..span_name import SpanName
from .base import apply_attributes, reference_id_context

__all__ = [
    "AgentSpanSchema",
]


class AgentSpanSchema:
    """Schema for agent-related spans."""

    def __init__(
        self,
        tracer: Tracer,
        upsert_started_fn: Optional[Callable[[Span], bool]] = None,
    ):
        """Initialize agent span schema.

        Args:
            tracer: The OpenTelemetry tracer to use
            upsert_started_fn: Optional function to upsert span on start
        """
        self._tracer = tracer
        self._upsert_started = upsert_started_fn

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

        # Set reference_id in context for all child spans
        reference_id = agent_id
        token = reference_id_context.set(reference_id)

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
        )

        try:
            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,
            ) as span:
                apply_attributes(span, attrs)
                if self._upsert_started:
                    self._upsert_started(span)

                try:
                    yield span
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.set_attribute(
                        "error",
                        json.dumps({"message": str(e), "type": type(e).__name__}),
                    )
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        finally:
            # Reset context variable when agent run completes
            reference_id_context.reset(token)

    def emit_agent_output(self, output: Any) -> None:
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
            attrs = AgentOutputSpanAttributes(output=output_str)
            apply_attributes(span, attrs)
            span.set_status(Status(StatusCode.OK))
