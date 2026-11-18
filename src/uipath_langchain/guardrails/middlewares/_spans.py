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


def action_name(action: Any) -> str:
    """``BlockAction`` -> ``Block``."""
    name = type(action).__name__
    return name.removesuffix("Action") or name


@contextmanager
def guardrail_action_span(
    guardrail_name: str, action: Any, result: GuardrailValidationResult
) -> Generator[Span]:
    """Record the action taken on a failed ``result``."""
    severity = getattr(action, "severity_level", None)
    span = _new_span(
        action_name(action),
        GUARDRAIL_ACTION,
        guardrailName=guardrail_name,
        action=action_name(action),
        guardrailSpanId=result.span_id,
        reason=result.reason or None,
        severityLevel=(
            severity.name.title()
            if isinstance(severity, LoggingSeverityLevel)
            else None
        ),
    )
    with _active(span):
        yield span
