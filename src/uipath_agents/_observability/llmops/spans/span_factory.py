"""Span factory for LLMOps traces.

Creates typed OpenTelemetry spans matching UiPath schema.
Used by LlmOpsInstrumentationCallback to instrument LangGraph agents.
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, cast

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Span, Status, StatusCode
from uipath.tracing import SpanStatus

from .spans_schema import (
    AgentSpanSchema,
    GuardrailSpanSchema,
    LlmSpanSchema,
    SpanUpsertProtocol,
    SyntheticReadableSpan,
    ToolSpanSchema,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LlmOpsSpanFactory",
    "SyntheticReadableSpan",
    "SpanUpsertProtocol",
]


class LlmOpsSpanFactory:
    """Factory for creating LLMOps-schema OpenTelemetry spans.

    Key features:
    - Creates typed spans matching UiPath LLMOps schema
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

        # Initialize span schemas with upsert callbacks
        self._agent_schema = AgentSpanSchema(
            self._tracer,
            upsert_started_fn=self.upsert_span_started,
        )
        self._llm_schema = LlmSpanSchema(
            self._tracer,
            upsert_started_fn=self.upsert_span_started,
        )
        self._tool_schema = ToolSpanSchema(
            self._tracer,
            upsert_started_fn=self.upsert_span_started,
        )
        self._guardrail_schema = GuardrailSpanSchema(
            self._tracer,
            upsert_started_fn=self.upsert_span_started,
            upsert_complete_fn=self.upsert_span_complete,
        )

    # -------------------------------------------------------------------------
    # Agent spans
    # -------------------------------------------------------------------------

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
        with self._agent_schema.start_agent_run(
            agent_name,
            agent_id=agent_id,
            is_conversational=is_conversational,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_data=input_data,
            input_schema=input_schema,
            output_schema=output_schema,
            source=source,
        ) as span:
            yield span

    def emit_agent_output(self, output: Any) -> None:
        """Emit agent output span (short-lived, captures final output).

        Args:
            output: The agent's output (will be JSON serialized if dict/list)
        """
        self._agent_schema.emit_agent_output(output)

    # -------------------------------------------------------------------------
    # LLM spans
    # -------------------------------------------------------------------------

    def start_llm_call(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        input: Optional[str] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an LLM call span (outer wrapper for LLM iteration).

        Returns a span that must be explicitly ended.
        Use for callback-based instrumentation where context managers don't fit.

        Args:
            model_name: Name of the model being called
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            input: The user input/prompt for this LLM call
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        return self._llm_schema.start_llm_call(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            input=input,
            parent_span=parent_span,
        )

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
        return self._llm_schema.start_model_run(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            parent_span=parent_span,
        )

    # -------------------------------------------------------------------------
    # Tool spans
    # -------------------------------------------------------------------------

    def start_tool_call(
        self,
        tool_name: str,
        tool_type: str = "toolCall",
        tool_type_value: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a tool call span.

        Args:
            tool_name: Name of the tool being called
            tool_type: Span type string (toolCall, processTool, etc.)
            tool_type_value: Tool type for display (Agent, Process, Integration)
            arguments: Arguments passed to the tool
            call_id: LLM tool call ID
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        return self._tool_schema.start_tool_call(
            tool_name=tool_name,
            tool_type=tool_type,
            tool_type_value=tool_type_value,
            arguments=arguments,
            call_id=call_id,
            parent_span=parent_span,
        )

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

        Creates a span named after the action app for human-in-the-loop escalations.
        The span name reflects the escalation destination (e.g., "SimpleApprovalApp").

        Args:
            app_name: Name of the action center app (used as span name)
            arguments: Arguments passed to the escalation
            channel_type: Type of channel (e.g., "actionCenter")
            assignee: Who the task is assigned to
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        return self._tool_schema.start_escalation_tool(
            app_name,
            arguments=arguments,
            channel_type=channel_type,
            assignee=assignee,
            parent_span=parent_span,
        )

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
        return self._tool_schema.start_process_tool(
            process_name,
            arguments=arguments,
            parent_span=parent_span,
        )

    def start_agent_tool(
        self,
        agent_name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an agent tool span (child of tool call for agent-as-tool).

        Creates a span named after the invoked agent.
        The span name reflects which agent was called (e.g., "A_plus_B").

        Args:
            agent_name: Name of the agent (used as span name)
            arguments: Arguments passed to the agent
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        return self._tool_schema.start_agent_tool(
            agent_name,
            arguments=arguments,
            parent_span=parent_span,
        )

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
        return self._tool_schema.start_integration_tool(
            tool_name,
            parent_span=parent_span,
        )

    # -------------------------------------------------------------------------
    # Guardrail spans
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
        return self._guardrail_schema.start_guardrails_container(
            scope=scope,
            stage=stage,
            parent_span=parent_span,
        )

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
        return self._guardrail_schema.start_guardrail_evaluation(
            guardrail_name=guardrail_name,
            guardrail_description=guardrail_description,
            scope=scope,
            parent_span=parent_span,
        )

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
        self._guardrail_schema.end_guardrail_evaluation(
            span=span,
            validation_passed=validation_passed,
            validation_result=validation_result,
            action=action,
            severity_level=severity_level,
            reason=reason,
        )

    def start_guardrail_escalation(
        self,
        guardrail_name: str,
        scope: str,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a guardrail escalation span ('Review task').

        Creates a 'Review task' span as a child of the guardrail evaluation span.
        Tracks the human review lifecycle for escalated guardrail violations.

        Args:
            guardrail_name: Name of the guardrail that escalated
            scope: Scope of guardrail ("agent", "llm", "tool")
            parent_span: The guardrail evaluation span (parent)

        Returns:
            The started Span (caller must track for later completion)
        """
        return self._guardrail_schema.start_guardrail_escalation(
            guardrail_name=guardrail_name,
            scope=scope,
            parent_span=parent_span,
        )

    def end_guardrail_escalation(
        self,
        span: Span,
        review_outcome: str,
        reviewed_by: Optional[str] = None,
        review_reason: Optional[Any] = None,
        reviewed_inputs: Optional[Any] = None,
        reviewed_outputs: Optional[Any] = None,
    ) -> None:
        """End a guardrail escalation span with review results.

        Args:
            span: The escalation span to end
            review_outcome: "Approved" or "Rejected"
            reviewed_by: Who completed the review
            review_reason: Reason for the decision
            reviewed_inputs: Modified inputs (if any)
            reviewed_outputs: Modified outputs (if any)
        """
        self._guardrail_schema.end_guardrail_escalation(
            span=span,
            review_outcome=review_outcome,
            reviewed_by=reviewed_by,
            review_reason=review_reason,
            reviewed_inputs=reviewed_inputs,
            reviewed_outputs=reviewed_outputs,
        )

    # -------------------------------------------------------------------------
    # Span lifecycle helpers
    # -------------------------------------------------------------------------

    def end_span_ok(self, span: Span) -> None:
        """End a span with OK status and upsert final state."""
        span.set_status(Status(StatusCode.OK))
        span.end()
        self.upsert_span_complete(span, status=SpanStatus.OK)

    def end_span_error(self, span: Span, error: Exception) -> None:
        """End a span with ERROR status and upsert final state."""
        span.set_attribute(
            "error",
            json.dumps({"message": str(error), "type": type(error).__name__}),
        )
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()
        self.upsert_span_complete(span, status=SpanStatus.ERROR)

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

    def upsert_span_suspended(self, span: Span) -> bool:
        """Upsert span to backend with UNSET status (no end time).

        Used when suspending interruptible tools. Sends span state to backend
        so it persists across process restart. UNSET status with no end time
        indicates the span is in-progress or waiting for continuation.

        Args:
            span: The span to upsert

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

            result = self._exporter.upsert_span(span, status_override=SpanStatus.UNSET)
            success = result == SpanExportResult.SUCCESS
            if success:
                logger.debug(
                    "Upserted span %s with UNSET status (suspended)",
                    span.name,
                )
            else:
                logger.warning("Failed to upsert span %s", span.name)
            return success
        except Exception:
            logger.exception("Error upserting span with UNSET status")
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
        span_data: Dict[str, Any],
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
