"""Manual instrumentation tracer for UiPath Agents.

Creates OpenTelemetry spans with UiPath schema attributes.
Used by UiPathTracingCallback to instrument LangGraph agents.
"""

import json
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from .schema import SpanName
from .schema import SpanType as SpanTypeEnum
from .span_attributes import (
    AgentOutputSpanAttributes,
    AgentRunSpanAttributes,
    BaseSpanAttributes,
    CompletionSpanAttributes,
    LlmCallSpanAttributes,
    ModelSettings,
    ToolCallSpanAttributes,
)


class UiPathTracer:
    """Manual tracer creating UiPath-schema OpenTelemetry spans.

    Key features:
    - Creates exact UiPath schema spans (type, agentName, etc.)
    - Supports immediate span emission (on start, not just end)
    - Correct parent-child hierarchy via OpenTelemetry context
    - Uses typed attribute classes for type safety
    """

    def __init__(self, tracer_name: str = "uipath-agents", version: str = "1.0.0"):
        """Initialize the tracer.

        Args:
            tracer_name: Name for the OpenTelemetry tracer
            version: Version string for the tracer
        """
        self._tracer = trace.get_tracer(tracer_name, version)

    @staticmethod
    def _apply_attributes(span: Span, attrs: BaseSpanAttributes) -> None:
        """Apply typed attributes to an OpenTelemetry span.

        OTEL only accepts primitives. Complex objects (dict/list) are JSON
        serialized here. To avoid double-escaping in storage, the span
        processor should handle final serialization.

        Args:
            span: The span to set attributes on
            attrs: Typed attributes to apply
        """
        for key, value in attrs.to_otel_attributes().items():
            if value is None:
                continue
            # OTEL only accepts primitives - serialize complex objects
            if isinstance(value, (dict, list)):
                span.set_attribute(key, json.dumps(value))
            else:
                span.set_attribute(key, value)

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

        Yields:
            The OpenTelemetry Span object
        """
        span_name = SpanName.agent_run(agent_name, is_conversational)

        # Create typed attributes
        attrs = AgentRunSpanAttributes(
            agent_name=agent_name,
            agent_id=agent_id,
            is_conversational=is_conversational if is_conversational else None,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input=input_data,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            self._apply_attributes(span, attrs)

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

    def start_llm_call(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an LLM call span (outer wrapper for LLM iteration).

        Returns a span that must be explicitly ended.
        Use for callback-based instrumentation where context managers don't fit.

        Args:
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = self._tracer.start_span(
            SpanName.LLM_CALL,
            kind=SpanKind.INTERNAL,
            context=context,
        )
        # LLM call: type=llmCall, no model (model only on child Model run span)
        settings = None
        if max_tokens is not None or temperature is not None:
            settings = ModelSettings(max_tokens=max_tokens, temperature=temperature)
        attrs = LlmCallSpanAttributes(settings=settings)
        self._apply_attributes(span, attrs)
        return span

    def start_model_run(
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a model run span (inner actual API call).

        Should be a child of an LLM call span.

        Args:
            model_name: Name of the model being called
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = self._tracer.start_span(
            SpanName.MODEL_RUN,
            kind=SpanKind.INTERNAL,
            context=context,
        )
        # Model run: type="completion", has model and nested settings
        settings = None
        if max_tokens is not None or temperature is not None:
            settings = ModelSettings(max_tokens=max_tokens, temperature=temperature)
        attrs = CompletionSpanAttributes(
            model=model_name,
            settings=settings,
        )
        self._apply_attributes(span, attrs)
        return span

    def start_tool_call(
        self,
        tool_name: str,
        tool_type: SpanTypeEnum = SpanTypeEnum.TOOL_CALL,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a tool call span.

        Args:
            tool_name: Name of the tool being called
            tool_type: Type of tool (TOOL_CALL, PROCESS_TOOL, etc.)
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = self._tracer.start_span(
            SpanName.tool_call(tool_name),
            kind=SpanKind.INTERNAL,
            context=context,
        )
        # Use typed attributes - pass span_type to override the type field
        attrs = ToolCallSpanAttributes(tool_name=tool_name, span_type=tool_type.value)
        self._apply_attributes(span, attrs)
        return span

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
            # Use typed attributes
            attrs = AgentOutputSpanAttributes(output=output_str)
            self._apply_attributes(span, attrs)
            span.set_status(Status(StatusCode.OK))

    @staticmethod
    def end_span_ok(span: Span) -> None:
        """End a span with OK status."""
        span.set_status(Status(StatusCode.OK))
        span.end()

    @staticmethod
    def end_span_error(span: Span, error: Exception) -> None:
        """End a span with ERROR status."""
        span.set_attribute(
            "error",
            json.dumps({"message": str(error), "type": type(error).__name__}),
        )
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()
