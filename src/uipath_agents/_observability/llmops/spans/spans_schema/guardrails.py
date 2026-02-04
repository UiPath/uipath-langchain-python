"""Guardrail span schemas.

Handles guardrail container, evaluation, and escalation spans.
"""

import ast
import json
from typing import Any, Callable, Optional

from opentelemetry.trace import (
    Span,
    SpanKind,
    Tracer,
)
from uipath.core.guardrails import GuardrailScope
from uipath_langchain.agent.guardrails.types import ExecutionStage

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
from .base import apply_attributes, create_span, end_span_error, end_span_ok

__all__ = [
    "GuardrailSpanSchema",
]


def to_json_string(value: Any) -> str:
    """Convert a value to a JSON string, handling Python repr format."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed)
        except (ValueError, SyntaxError):
            pass
        return value
    return str(value)


def _format_guardrail_payload(payload: Optional[Any]) -> Optional[str]:
    """Format payload for span attribute, handling None values.

    Payload can contain input and/or output; if any of input or output is None,
    it will be excluded. If only one non-null value remains, use that directly.

    Args:
        payload: Dict with input/output keys, or any other value

    Returns:
        JSON string representation of the payload, or None if empty
    """
    if not payload:
        return None

    non_null = {k: v for k, v in payload.items() if v is not None}
    if len(non_null) == 1:
        payload_attr = next(iter(non_null.values()))
    else:
        payload_attr = non_null

    if payload_attr is not None:
        return to_json_string(payload_attr)
    return None


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
        # Select appropriate attributes class based on scope and stage
        attrs: BaseSpanAttributes
        if scope == GuardrailScope.AGENT:
            if stage == ExecutionStage.PRE_EXECUTION:
                attrs = AgentPreGuardrailsSpanAttributes()
                span_name = SpanName.AGENT_PRE_GUARDRAILS
            else:
                attrs = AgentPostGuardrailsSpanAttributes()
                span_name = SpanName.AGENT_POST_GUARDRAILS
        elif scope == GuardrailScope.LLM:
            if stage == ExecutionStage.PRE_EXECUTION:
                attrs = LlmPreGuardrailsSpanAttributes()
                span_name = SpanName.LLM_PRE_GUARDRAILS
            else:
                attrs = LlmPostGuardrailsSpanAttributes()
                span_name = SpanName.LLM_POST_GUARDRAILS
        else:  # tool
            if stage == ExecutionStage.PRE_EXECUTION:
                attrs = ToolPreGuardrailsSpanAttributes()
                span_name = SpanName.TOOL_PRE_GUARDRAILS
            else:
                attrs = ToolPostGuardrailsSpanAttributes()
                span_name = SpanName.TOOL_POST_GUARDRAILS

        span = create_span(
            self._tracer,
            span_name,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

    def start_guardrail_evaluation(
        self,
        guardrail_name: str,
        guardrail_action: str,
        guardrail_description: Optional[str] = None,
        rule_details: Optional[list[str]] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an individual guardrail evaluation span.

        Args:
            guardrail_name: Name of the guardrail being evaluated
            guardrail_description: Optional description of the guardrail
            guardrail_action: "Log", "Block", "Filter" or "Escalate" - action that was configured to be enforced ig guardrail validation fails
            rule_details: Optional details about the guardrail rules being evaluated
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
            guardrail_action=guardrail_action,
            details=rule_details,
        )

        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

    def end_guardrail_evaluation(
        self,
        span: Span,
        validation_result: Optional[str] = None,
        action: Optional[str] = None,
        severity_level: Optional[str] = None,
        reason: Optional[str] = None,
        payload: Optional[Any] = None,
        excluded_fields: Optional[Any] = None,
        updated_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """End a guardrail evaluation span with result attributes.

        Args:
            span: The guardrail evaluation span to end
            validation_result: The validation result message (if failed)
            action: The action taken ("allow", "block", "log", "escalate")
            severity_level: Severity level for log actions (for Log action)
            reason: Reason for block/skip actions (for Block action)
            payload: Data was validated against the guardrail rule
            excluded_fields: List of fields to exclude from the guardrail evaluation (for Filter action)
            updated_data: Optional updated data (contains updated input and updated output) (for Filter action)
        """
        if validation_result:
            span.set_attribute("validationResult", validation_result)
        if action:
            span.set_attribute("action", action)
        if severity_level:
            span.set_attribute("severityLevel", severity_level.title())
        if reason:
            span.set_attribute("reason", reason)
        if excluded_fields:
            span.set_attribute("excludedFields", to_json_string(excluded_fields))
        if updated_data and updated_data.get("input"):
            span.set_attribute(
                "updatedInput", to_json_string(updated_data.get("input"))
            )
        if updated_data and updated_data.get("output"):
            span.set_attribute(
                "updatedOutput", to_json_string(updated_data.get("output"))
            )
        if payload:
            formatted_payload = _format_guardrail_payload(payload)
            if formatted_payload:
                span.set_attribute("payload", formatted_payload)

        # Guardrail failure != span error
        end_span_ok(span, self._upsert_complete)

    def upsert_guardrail_evaluation(
        self,
        span: Span,
        validation_result: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:
        """Update a guardrail evaluation span with attributes and upsert without ending.

        Used to update span attributes while the span is still in progress
        (e.g., before HITL escalation). Upserts with UNSET status so the span
        appears in traces as "in progress".

        Args:
            span: The guardrail evaluation span to update
            validation_result: The validation result message
            payload: Data that was validated against the guardrail rule
        """
        if validation_result:
            span.set_attribute("validationResult", validation_result)
        if payload:
            formatted_payload = _format_guardrail_payload(payload)
            if formatted_payload:
                span.set_attribute("payload", formatted_payload)

        # Upsert without ending (UNSET status = in progress)
        if self._upsert_started:
            self._upsert_started(span)

    def error_guardrail_evaluation(
        self,
        span: Span,
        error_message: str,
        validation_result: Optional[str] = None,
        action: Optional[str] = None,
        reason: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:
        """End a guardrail evaluation span with result attributes and error status.

        Args:
            span: The guardrail evaluation span to end
            validation_result: The validation result message
            action: The action taken ("allow", "block", "log", "escalate")
            reason: Reason for block/skip actions (for Block action)
            payload: Data was validated against the guardrail rule
            error_message: Error message
        """
        if validation_result:
            span.set_attribute("validationResult", validation_result)
        if action:
            span.set_attribute("action", action)
        if reason:
            span.set_attribute("reason", reason)
        if payload:
            formatted_payload = _format_guardrail_payload(payload)
            if formatted_payload:
                span.set_attribute("payload", formatted_payload)

        error = Exception(str(error_message))
        end_span_error(span, error, self._upsert_complete)

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
