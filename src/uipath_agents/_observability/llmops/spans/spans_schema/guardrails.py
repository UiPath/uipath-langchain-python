"""Guardrail span schemas.

Handles guardrail container, evaluation, and escalation spans.
"""

import json
from typing import Any, Callable, Optional

from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
)

from ..span_attributes import (
    AgentPostGuardrailsSpanAttributes,
    AgentPreGuardrailsSpanAttributes,
    BaseSpanAttributes,
    GuardrailEscalationSpanAttributes,
    GuardrailEvaluationSpanAttributes,
    LlmPostGuardrailsSpanAttributes,
    LlmPreGuardrailsSpanAttributes,
    ToolPostGuardrailsSpanAttributes,
    ToolPreGuardrailsSpanAttributes,
)
from ..span_name import SpanName
from .base import apply_attributes, create_span, end_span_ok

__all__ = [
    "GuardrailSpanSchema",
]


class GuardrailSpanSchema:
    """Schema for guardrail-related spans."""

    def __init__(
        self,
        tracer: Tracer,
        upsert_started_fn: Optional[Callable[[Span], bool]] = None,
        upsert_complete_fn: Optional[Callable[[Span, int], bool]] = None,
    ):
        """Initialize guardrail span schema.

        Args:
            tracer: The OpenTelemetry tracer to use
            upsert_started_fn: Optional function to upsert span on start
            upsert_complete_fn: Optional function to upsert span on complete
        """
        self._tracer = tracer
        self._upsert_started = upsert_started_fn
        self._upsert_complete = upsert_complete_fn

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
        span_name = SpanName.guardrails_container(scope, stage)
        span = create_span(
            self._tracer,
            span_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
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

        apply_attributes(span, attrs)
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
        span_name = SpanName.guardrail(guardrail_name)
        span = create_span(
            self._tracer,
            span_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )

        attrs = GuardrailEvaluationSpanAttributes(
            guardrail_name=guardrail_name,
            guardrail_description=guardrail_description,
        )

        apply_attributes(span, attrs)
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
            span.set_attribute("severityLevel", severity_level.title())
        if reason:
            span.set_attribute("reason", reason)

        # Guardrail failure != span error
        span.set_status(Status(StatusCode.OK))
        span.end()

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
        span = create_span(
            self._tracer,
            SpanName.REVIEW_TASK,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )

        attrs = GuardrailEscalationSpanAttributes(
            guardrail_name=guardrail_name,
            review_status="waiting",
        )

        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

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
        span.set_attribute("reviewStatus", "completed")
        span.set_attribute("reviewOutcome", review_outcome)
        if reviewed_by:
            span.set_attribute("reviewedBy", reviewed_by)
        if review_reason:
            if isinstance(review_reason, dict):
                span.set_attribute("reviewReason", json.dumps(review_reason))
            else:
                span.set_attribute("reviewReason", str(review_reason))
        if reviewed_inputs:
            if isinstance(reviewed_inputs, dict):
                span.set_attribute("reviewedInputs", json.dumps(reviewed_inputs))
            else:
                span.set_attribute("reviewedInputs", str(reviewed_inputs))
        if reviewed_outputs:
            if isinstance(reviewed_outputs, dict):
                span.set_attribute("reviewedOutputs", json.dumps(reviewed_outputs))
            else:
                span.set_attribute("reviewedOutputs", str(reviewed_outputs))

        end_span_ok(span, self._upsert_complete)
