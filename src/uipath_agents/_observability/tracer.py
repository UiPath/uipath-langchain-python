"""Manual instrumentation tracer for UiPath Agents.

Creates OpenTelemetry spans with UiPath schema attributes.
Used by UiPathTracingCallback to instrument LangGraph agents.
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Protocol, cast

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import (
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)
from uipath.tracing import SpanStatus

from .schema import SpanName
from .schema import SpanType as SpanTypeEnum
from .span_attributes import (
    AgentOutputSpanAttributes,
    AgentPostGuardrailsSpanAttributes,
    AgentPreGuardrailsSpanAttributes,
    AgentRunSpanAttributes,
    BaseSpanAttributes,
    CompletionSpanAttributes,
    EscalationToolSpanAttributes,
    GuardrailEvaluationSpanAttributes,
    IntegrationToolSpanAttributes,
    LlmCallSpanAttributes,
    LlmPostGuardrailsSpanAttributes,
    LlmPreGuardrailsSpanAttributes,
    ModelSettings,
    ProcessToolSpanAttributes,
    ToolCallSpanAttributes,
    ToolPostGuardrailsSpanAttributes,
    ToolPreGuardrailsSpanAttributes,
    get_agent_version,
    get_execution_type,
)

logger = logging.getLogger(__name__)


class SyntheticReadableSpan:
    """Minimal ReadableSpan for upsert from saved data after process restart."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: Dict[str, Any],
        parent_span_id: Optional[str] = None,
    ):
        trace_id_int = int(trace_id, 16)
        span_id_int = int(span_id, 16)
        parent_id_int = int(parent_span_id, 16) if parent_span_id else None

        self.name = name
        self.start_time = start_time_ns
        self.end_time = end_time_ns
        self.attributes = attributes
        self.status = Status(StatusCode.OK)
        self.kind = SpanKind.INTERNAL
        self.events: tuple[Any, ...] = ()
        self.links: tuple[Any, ...] = ()
        self.resource = None
        self.instrumentation_info = None
        self.parent = (
            SpanContext(
                trace_id=trace_id_int,
                span_id=parent_id_int,
                is_remote=False,
                trace_flags=TraceFlags(0x01),
            )
            if parent_id_int
            else None
        )
        self._span_context = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )

    def get_span_context(self) -> SpanContext:
        return self._span_context


class SpanUpsertProtocol(Protocol):
    """Protocol for span upsert operations."""

    def upsert_span(
        self,
        span: ReadableSpan,
        status_override: Optional[int] = None,
    ) -> SpanExportResult: ...


class UiPathTracer:
    """Manual tracer creating UiPath-schema OpenTelemetry spans.

    Key features:
    - Creates exact UiPath schema spans (type, agentName, etc.)
    - Supports immediate span emission (on start, not just end)
    - Correct parent-child hierarchy via OpenTelemetry context
    - Uses typed attribute classes for type safety
    """

    def __init__(
        self,
        tracer_name: str = "uipath-agents",
        version: str = "1.0.0",
        exporter: Optional[SpanUpsertProtocol] = None,
    ):
        """Initialize the tracer.

        Args:
            tracer_name: Name for the OpenTelemetry tracer
            version: Version string for the tracer
            exporter: Optional exporter for upsert operations (interruptible spans)
        """
        self._tracer = trace.get_tracer(tracer_name, version)
        self._exporter = exporter

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
            is_conversational=is_conversational,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input=input_data,
            input_schema=input_schema,
            output_schema=output_schema,
            execution_type=get_execution_type(),
            agent_version=get_agent_version(),
            reference_id=agent_id,
        )

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            self._apply_attributes(span, attrs)
            self.upsert_span_started(span)

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
        input: Optional[str] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an LLM call span (outer wrapper for LLM iteration).

        Returns a span that must be explicitly ended.
        Use for callback-based instrumentation where context managers don't fit.

        Args:
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            input: The user input/prompt for this LLM call
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
        attrs = LlmCallSpanAttributes(settings=settings, input=input)
        self._apply_attributes(span, attrs)
        self.upsert_span_started(span)
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
        self.upsert_span_started(span)
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
        self.upsert_span_started(span)
        return span

    def start_escalation_tool(
        self,
        app_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        channel_type: Optional[str] = None,
        assignee: Optional[str] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an escalation tool span (child of tool call).

        Creates a span named after the action app for HITL escalations.
        This matches the Temporal pattern where tool call spans have
        a child span named after the app (e.g., "SimpleApprovalApp").

        Args:
            app_name: Name of the action center app (used as span name)
            arguments: Arguments passed to the escalation
            channel_type: Type of channel (e.g., "actionCenter")
            assignee: Who the task is assigned to
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = self._tracer.start_span(
            app_name,
            kind=SpanKind.INTERNAL,
            context=context,
        )

        attrs = EscalationToolSpanAttributes(
            arguments=arguments,
            channel_type=channel_type,
            assigned_to=assignee,
        )
        self._apply_attributes(span, attrs)
        self.upsert_span_started(span)
        return span

    def start_process_tool(
        self,
        process_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a process tool span (child of tool call).

        Creates a span named after the process for interruptible process calls.
        This matches the pattern where tool call spans have a child span
        named after the process (e.g., "InvoiceProcessor").

        Args:
            process_name: Name of the UiPath process (used as span name)
            arguments: Arguments passed to the process
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = self._tracer.start_span(
            process_name,
            kind=SpanKind.INTERNAL,
            context=context,
        )

        attrs = ProcessToolSpanAttributes(
            tool_name=process_name,
            arguments=arguments,
        )
        self._apply_attributes(span, attrs)
        self.upsert_span_started(span)
        return span

    def start_integration_tool(
        self,
        tool_name: str,
        *,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an integration tool span (child of tool call).

        Creates a child span for integration tool execution. This replaces
        the SDK's activity_invoke span which gets filtered out from LLMOps.

        Args:
            tool_name: Name of the integration tool (used as span name)
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = self._tracer.start_span(
            tool_name,
            kind=SpanKind.INTERNAL,
            context=context,
        )

        attrs = IntegrationToolSpanAttributes(tool_name=tool_name)
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

    # -------------------------------------------------------------------------
    # UpsertSpan methods for interruptible spans
    # -------------------------------------------------------------------------

    def upsert_span_started(self, span: Span) -> bool:
        """Upsert span on start with UNSET status for live visibility.

        Enables live updates - UI shows spans immediately as they start,
        not just when they complete.

        Args:
            span: The span to upsert

        Returns:
            True if upsert succeeded, False otherwise
        """
        return self.upsert_span_complete(span, status=SpanStatus.UNSET)

    def upsert_span_running(self, span: Span) -> bool:
        """Upsert span to backend with RUNNING status.

        Used when suspending interruptible tools. Sends span state to backend
        so it survives process restart.

        Args:
            span: The span to upsert

        Returns:
            True if upsert succeeded, False otherwise
        """
        if not self._exporter:
            logger.debug("No exporter configured, skipping upsert")
            return False

        try:
            # Span must be ReadableSpan for exporter
            if not isinstance(span, ReadableSpan):
                logger.warning("Span is not ReadableSpan, cannot upsert")
                return False

            result = self._exporter.upsert_span(
                span, status_override=SpanStatus.RUNNING
            )
            success = result == SpanExportResult.SUCCESS
            if success:
                logger.debug("Upserted span %s with RUNNING status", span.name)
            else:
                logger.warning("Failed to upsert span %s", span.name)
            return success
        except Exception:
            logger.exception("Error upserting span with RUNNING status")
            return False

    def upsert_span_complete(self, span: Span, status: int = SpanStatus.OK) -> bool:
        """Upsert span to backend with final status.

        Used when resuming interruptible tools. Sends final state to backend
        with correct end_time (capturing full wait duration).

        Args:
            span: The span to upsert
            status: Final status (OK, ERROR, etc.)

        Returns:
            True if upsert succeeded, False otherwise
        """
        if not self._exporter:
            logger.debug("No exporter configured, skipping upsert")
            return False

        try:
            if not isinstance(span, ReadableSpan):
                logger.warning("Span is not ReadableSpan, cannot upsert")
                return False

            result = self._exporter.upsert_span(span, status_override=status)
            success = result == SpanExportResult.SUCCESS
            if success:
                logger.debug("Upserted span %s with status %d", span.name, status)
            else:
                logger.warning("Failed to upsert span %s with final status", span.name)
            return success
        except Exception:
            logger.exception("Error upserting span with final status")
            return False

    def upsert_span_complete_by_data(
        self,
        trace_id: str,
        span_data: Dict[str, Any],  # PendingSpanData or similar dict
        status: int = SpanStatus.OK,
    ) -> bool:
        """Upsert span to backend from saved data.

        Used after process restart when original span objects are lost.
        Creates a synthetic span from saved data and upserts with final status.

        Args:
            trace_id: Hex-encoded trace ID
            span_data: Saved span data (span_id, name, start_time_ns, attributes)
            status: Final status (OK, ERROR, etc.)

        Returns:
            True if upsert succeeded, False otherwise
        """
        if not self._exporter:
            logger.debug("No exporter configured, skipping upsert")
            return False

        try:
            synthetic_span = SyntheticReadableSpan(
                trace_id=trace_id,
                span_id=span_data.get("span_id", "0" * 16),
                name=span_data.get("name", "unknown"),
                start_time_ns=span_data.get("start_time_ns", 0),
                end_time_ns=time.time_ns(),
                attributes=span_data.get("attributes", {}),
                parent_span_id=span_data.get("parent_span_id"),
            )

            # Cast to ReadableSpan - SyntheticReadableSpan implements required interface
            result = self._exporter.upsert_span(
                cast(ReadableSpan, synthetic_span), status_override=status
            )
            success = result == SpanExportResult.SUCCESS
            if success:
                logger.debug(
                    "Upserted synthetic span %s with status %d",
                    span_data.get("name", "unknown"),
                    status,
                )
            else:
                logger.warning(
                    "Failed to upsert synthetic span %s",
                    span_data.get("name", "unknown"),
                )
            return success
        except Exception:
            logger.exception("Error upserting span from saved data")
            return False

    # -------------------------------------------------------------------------
    # Guardrail Spans
    # -------------------------------------------------------------------------

    def start_guardrails_container(
        self,
        scope: str,
        stage: str,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a container span for a guardrails phase.

        Args:
            scope: "agent", "llm", or "tool"
            stage: "pre" or "post"
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started container Span (caller must call span.end())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span_name = SpanName.guardrails_container(scope, stage)
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.INTERNAL,
            context=context,
        )

        # Select appropriate attributes class based on scope and stage
        attrs: BaseSpanAttributes
        if scope == "agent":
            if stage == "pre":
                attrs = AgentPreGuardrailsSpanAttributes()
            else:
                attrs = AgentPostGuardrailsSpanAttributes()
        elif scope == "llm":
            if stage == "pre":
                attrs = LlmPreGuardrailsSpanAttributes()
            else:
                attrs = LlmPostGuardrailsSpanAttributes()
        else:  # tool
            if stage == "pre":
                attrs = ToolPreGuardrailsSpanAttributes()
            else:
                attrs = ToolPostGuardrailsSpanAttributes()

        self._apply_attributes(span, attrs)
        return span

    def start_guardrail_evaluation(
        self,
        guardrail_name: str,
        guardrail_description: Optional[str] = None,
        scope: str = "agent",
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an individual guardrail evaluation span.

        Args:
            guardrail_name: Name of the guardrail being evaluated
            guardrail_description: Optional description of the guardrail
            scope: "agent", "llm", or "tool" - determines span type
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started evaluation Span (caller must call end_guardrail_evaluation())
        """
        parent = parent_span or trace.get_current_span()
        context = trace.set_span_in_context(parent) if parent else None

        span_name = SpanName.guardrail(guardrail_name)
        span = self._tracer.start_span(
            span_name,
            kind=SpanKind.INTERNAL,
            context=context,
        )

        attrs = GuardrailEvaluationSpanAttributes(
            guardrail_name=guardrail_name,
            guardrail_description=guardrail_description,
        )

        self._apply_attributes(span, attrs)
        return span

    def end_guardrail_evaluation(
        self,
        span: Span,
        validation_passed: bool,
        validation_result: Optional[str] = None,
        action: Optional[str] = None,
        severity_level: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """End a guardrail evaluation span with result attributes.

        Args:
            span: The guardrail evaluation span to end
            validation_passed: Whether the guardrail validation passed
            validation_result: The validation result message (if failed)
            action: The action taken ("allow", "block", "log", "escalate")
            severity_level: Severity level for log actions
            reason: Reason for block/skip actions
        """
        if validation_result:
            span.set_attribute("validationResult", validation_result)
        if action:
            span.set_attribute("guardrailAction", action)
            span.set_attribute("action", action)
        if severity_level:
            span.set_attribute("severityLevel", severity_level)
        if reason:
            span.set_attribute("reason", reason)

        if validation_passed:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.OK))  # Guardrail failure != span error

        span.end()
