"""Manual instrumentation tracer matching Temporal implementation pattern.

This module provides the UiPathTracer class that creates OpenTelemetry spans
with the exact schema matching Temporal implementation backend.
"""
import json
import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from .schema import SpanName, SpanType


def is_custom_instrumentation_enabled() -> bool:
    """Check if custom instrumentation is enabled via environment variable.

    Returns:
        True if UIPATH_CUSTOM_INSTRUMENTATION env var is set to "true" (case-insensitive)
    """
    return os.getenv("UIPATH_CUSTOM_INSTRUMENTATION", "").lower() == "true"


class UiPathTracer:
    """Manual tracer matching Temporal implementation TraceSpan pattern.

    Key differences from OpenInference auto-instrumentation:
    - Creates exact UiPath schema spans (type, agentName, etc.)
    - No noise spans (init, route_agent, terminate)
    - Correct parent-child hierarchy
    """

    def __init__(self, tracer_name: str = "uipath-langchain", version: str = "1.0.0"):
        """Initialize the tracer.

        Args:
            tracer_name: Name for the OpenTelemetry tracer
            version: Version string for the tracer
        """
        self._tracer = trace.get_tracer(tracer_name, version)

    @contextmanager
    def start_agent_run(
        self,
        agent_name: str,
        *,
        agent_id: Optional[str] = None,
        is_conversational: bool = False,
    ) -> Generator[Span, None, None]:
        """Start an agent run span.

        This is the root span for an agent execution. All LLM calls,
        tool calls, and guardrail checks should be children of this span.

        Args:
            agent_name: Name of the agent (used in span name)
            agent_id: Unique identifier for the agent instance
            is_conversational: Whether this is a conversational agent

        Yields:
            The OpenTelemetry Span object
        """
        span_name = SpanName.agent_run(agent_name, is_conversational)

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", SpanType.AGENT_RUN.value)
            span.set_attribute("agentName", agent_name)
            if agent_id:
                span.set_attribute("agentId", agent_id)
            span.set_attribute("source", "langchain")
            if is_conversational:
                span.set_attribute("isConversational", True)

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

    @contextmanager
    def start_llm_call(self) -> Generator[Span, None, None]:
        """Start an LLM call span (outer wrapper).

        This is the outer span that wraps an LLM iteration. The actual
        model API call should be wrapped in start_model_run() as a child.

        Hierarchy: Agent run → LLM call → Model run

        Yields:
            The OpenTelemetry Span object
        """
        with self._tracer.start_as_current_span(
            SpanName.LLM_CALL,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", SpanType.COMPLETION.value)

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

    @contextmanager
    def start_model_run(
        self,
        model_name: str,
    ) -> Generator[Span, None, None]:
        """Start a model run span (inner actual API call).

        This span represents the actual LLM API invocation. Should be a child
        of an LLM call span.

        Hierarchy: Agent run → LLM call → Model run

        Args:
            model_name: Name of the model being called (e.g., "gpt-4")

        Yields:
            The OpenTelemetry Span object
        """
        with self._tracer.start_as_current_span(
            SpanName.MODEL_RUN,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", SpanType.LLM_CALL.value)
            span.set_attribute("model", model_name)

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

    @contextmanager
    def start_tool_call(
        self,
        tool_name: str,
        tool_type: SpanType = SpanType.TOOL_CALL,
    ) -> Generator[Span, None, None]:
        """Start a tool call span.

        This span represents a tool invocation. Should be a child
        of an LLM call span (tools are called after LLM decides to use them).

        Hierarchy: Agent run → LLM call → Tool call

        Args:
            tool_name: Name of the tool being called
            tool_type: Type of tool (TOOL_CALL, PROCESS_TOOL, INTEGRATION_TOOL, etc.)

        Yields:
            The OpenTelemetry Span object
        """
        with self._tracer.start_as_current_span(
            SpanName.tool_call(tool_name),
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", tool_type.value)
            span.set_attribute("toolName", tool_name)

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

    def emit_agent_output(self, output: Any) -> None:
        """Emit agent output span.

        This is a short-lived span that captures the final agent output.
        Should be called at the end of agent execution.

        Args:
            output: The agent's output (will be JSON serialized if dict/list)
        """
        with self._tracer.start_as_current_span(
            SpanName.AGENT_OUTPUT,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("type", SpanType.AGENT_OUTPUT.value)
            if isinstance(output, (dict, list)):
                span.set_attribute("output", json.dumps(output))
            else:
                span.set_attribute("output", str(output) if output is not None else "")
            span.set_status(Status(StatusCode.OK))


# Global tracer instance (lazy initialization)
_tracer: Optional[UiPathTracer] = None


def get_tracer() -> UiPathTracer:
    """Get the global UiPathTracer instance.

    Returns:
        The singleton UiPathTracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = UiPathTracer()
    return _tracer


def reset_tracer() -> None:
    """Reset the global tracer instance.

    Primarily used for testing to ensure clean state.
    """
    global _tracer
    _tracer = None
