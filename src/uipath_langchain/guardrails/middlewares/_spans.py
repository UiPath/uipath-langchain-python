"""Guardrail spans shaped like low-code agents'."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from langgraph.errors import GraphBubbleUp
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode
from uipath.core.guardrails import GuardrailValidationResult
from uipath.core.tracing.span_utils import UiPathSpanUtils
from uipath.platform.guardrails.decorators import LoggingSeverityLevel
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

GUARDRAIL_ACTION = "guardrailAction"
GUARDRAIL_EVALUATION = "guardrailEvaluation"


def _new_span(name: str, span_type: str, **attributes: Any) -> Span:
    """Start a span under the deepest active span, as ``@traced`` does."""
    return trace.get_tracer(__name__).start_span(
        name,
        context=UiPathSpanUtils.get_parent_context(),
        attributes={
            "type": span_type,
            "span_type": span_type,
            "uipath.custom_instrumentation": True,
            **{key: value for key, value in attributes.items() if value is not None},
        },
    )


@contextmanager
def _active(span: Span) -> Generator[Span]:
    """Make ``span`` current and end it: OK, or ERROR on a failure (not interrupts)."""
    with trace.use_span(
        span, end_on_exit=True, record_exception=False, set_status_on_exception=False
    ):
        try:
            yield span
        except GraphBubbleUp:
            span.set_status(StatusCode.OK)
            raise
        except Exception as exc:
            span.set_status(StatusCode.ERROR, _error_text(exc))
            raise
        span.set_status(StatusCode.OK)


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, GuardrailBlockException):
        return f"Guardrail violation: {exc.detail}"
    return str(exc)


def span_guid(span: Span) -> str:
    """``span``'s id as a GUID, the form ``guardrailSpanId`` uses."""
    span_id = format(span.get_span_context().span_id, "016x")
    return f"00000000-0000-0000-{span_id[:4]}-{span_id[4:]}"


def action_name(action: Any) -> str:
    """``BlockAction`` -> ``Block``."""
    name = type(action).__name__
    return name.removesuffix("Action") or name


@contextmanager
def guardrail_action_span(
    guardrail_name: str,
    action: Any,
    result: GuardrailValidationResult,
    evaluation_span_id: str | None = None,
) -> Generator[Span]:
    """Record the action taken on a failed ``result``."""
    severity = getattr(action, "severity_level", None)
    span = _new_span(
        action_name(action),
        GUARDRAIL_ACTION,
        guardrailName=guardrail_name,
        action=action_name(action),
        guardrailSpanId=evaluation_span_id or result.span_id,
        reason=result.reason or None,
        severityLevel=(
            severity.name.title()
            if isinstance(severity, LoggingSeverityLevel)
            else None
        ),
    )
    with _active(span):
        yield span


@contextmanager
def guardrail_evaluation_span(
    guardrail_name: str, description: str | None = None
) -> Generator[Span]:
    """Record a rule check that runs in the agent."""
    span = _new_span(
        guardrail_name,
        GUARDRAIL_EVALUATION,
        guardrailName=guardrail_name,
        guardrailDescription=description,
        guardrailType="custom",
    )
    with _active(span):
        yield span


def set_validation_result(span: Span, result: GuardrailValidationResult) -> None:
    """Record ``result``'s outcome on its evaluation span."""
    span.set_attribute("validationResult", result.result.name)
    if result.reason:
        span.set_attribute("details", [result.reason])
