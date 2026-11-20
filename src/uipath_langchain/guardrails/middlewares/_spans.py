"""Guardrail spans shaped like low-code agents': actions, rule evaluations and
stage containers that group a stage's checks across separate middleware nodes."""

import logging
import threading
import time
import weakref
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeGuard

from langgraph.errors import GraphBubbleUp
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext, StatusCode
from uipath.core.guardrails import GuardrailScope, GuardrailValidationResult
from uipath.core.tracing.span_utils import UiPathSpanUtils
from uipath.platform.guardrails.decorators import (
    GuardrailExecutionStage,
    LoggingSeverityLevel,
)
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

GUARDRAIL_ACTION = "guardrailAction"
GUARDRAIL_EVALUATION = "guardrailEvaluation"
_CONTAINER_MARK = "uipath.guardrail_container"

logger = logging.getLogger(__name__)


def _new_span(
    name: str,
    span_type: str,
    context: otel_context.Context | None = None,
    **attributes: Any,
) -> Span:
    """Start a span under the deepest active span, as ``@traced`` does."""
    return trace.get_tracer(__name__).start_span(
        name,
        context=context or UiPathSpanUtils.get_parent_context(),
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


_PRE = GuardrailExecutionStage.PRE
_POST = GuardrailExecutionStage.POST

# (scope, stage) -> (container name, span type), as in low-code agents.
_STAGES: dict[tuple[GuardrailScope, GuardrailExecutionStage], tuple[str, str]] = {
    (GuardrailScope.AGENT, _PRE): ("Agent input guardrail check", "agentPreGuardrails"),
    (GuardrailScope.AGENT, _POST): (
        "Agent output guardrail check",
        "agentPostGuardrails",
    ),
    (GuardrailScope.LLM, _PRE): ("LLM input guardrail check", "llmPreGuardrails"),
    (GuardrailScope.LLM, _POST): ("LLM output guardrail check", "llmPostGuardrails"),
    (GuardrailScope.TOOL, _PRE): ("Tool input guardrail check", "toolPreGuardrails"),
    (GuardrailScope.TOOL, _POST): ("Tool output guardrail check", "toolPostGuardrails"),
}


class _GroupingState:
    """Process-wide state. Grouping stays off until a closer is registered."""

    def __init__(self) -> None:
        self.enabled = False
        self.in_use = False  # a guardrail middleware was built
        self.trace_manager: Any = None  # offered by the runtime factory
        self.closers: weakref.WeakKeyDictionary[Any, GuardrailStageCloser] = (
            weakref.WeakKeyDictionary()
        )  # trace manager -> its closer
        self.guardrail_nodes: dict[str, str] = {}  # node name -> stage span type
        self.containers: dict[tuple[int, str], Span] = {}  # (parent id, type)
        self.last_check_end: dict[int, int] = {}  # container span id -> ns
        self.lock = threading.Lock()


_state = _GroupingState()

_current_container: ContextVar[Span | None] = ContextVar(
    "uipath_guardrail_container", default=None
)
# Set while a container starts, so the closer skips it without taking the lock.
_starting_container: ContextVar[bool] = ContextVar(
    "uipath_guardrail_container_starting", default=False
)


def register_guardrail_node(
    node_name: str, scope: GuardrailScope, stage: GuardrailExecutionStage
) -> None:
    """Record which stage a guardrail hook's LangGraph node belongs to."""
    _state.guardrail_nodes[node_name] = _STAGES[(scope, stage)][1]
    guardrails_in_use()


def guardrails_in_use() -> None:
    """Called by guardrail middlewares when built; grouping needs one in use."""
    _state.in_use = True
    _activate()


@contextmanager
def guardrail_stage(
    scope: GuardrailScope | None, stage: GuardrailExecutionStage | None
) -> Generator[None]:
    """Run one check inside its stage's shared container."""
    try:
        container = _shared_container(scope, stage)
    except Exception:
        logger.warning("Could not group a guardrail check", exc_info=True)
        container = None
    if container is None:
        yield
        return
    token = _current_container.set(container)
    try:
        yield
    except GraphBubbleUp:
        raise
    except Exception as exc:
        container.set_status(StatusCode.ERROR, _error_text(exc))
        raise
    finally:
        _current_container.reset(token)
        with _state.lock:
            # A container closed meanwhile was already popped; don't re-add it.
            if any(open_ is container for open_ in _state.containers.values()):
                span_id = container.get_span_context().span_id
                _state.last_check_end[span_id] = time.time_ns()


def _shared_container(
    scope: GuardrailScope | None, stage: GuardrailExecutionStage | None
) -> Span | None:
    """Get or create the stage container: beside message nodes, inside ``tools``."""
    if not _state.enabled or (scope, stage) not in _STAGES:
        return None
    node = UiPathSpanUtils.get_external_current_span()
    if node is None or not node.is_recording():
        return None  # a sampled-out node never reports its end
    if scope == GuardrailScope.TOOL:
        parent: SpanContext | None = node.get_span_context()
    else:
        parent = getattr(node, "parent", None)
    if not _can_group_under(parent):
        return None

    name, span_type = _STAGES[(scope, stage)]  # type: ignore[index]
    key = (parent.span_id, span_type)
    previous: list[tuple[Span, int]] = []
    with _state.lock:
        container = _state.containers.get(key)
        if container is None:
            token = _starting_container.set(True)
            try:
                container = _new_span(
                    name,
                    span_type,
                    context=trace.set_span_in_context(NonRecordingSpan(parent)),
                    **{
                        _CONTAINER_MARK: True,
                        # Its parent is non-recording, so this isn't inherited.
                        "execution.id": (getattr(node, "attributes", None) or {}).get(
                            "execution.id"
                        ),
                    },
                )
            finally:
                _starting_container.reset(token)
            # No sibling may have started between the stages, e.g. when a
            # middleware answers a tool call without running the tool.
            previous = _pop_containers(
                lambda k: k[0] == parent.span_id and k[1] != span_type
            )
            _state.containers[key] = container
    _end_containers(previous, end_time=getattr(container, "start_time", None))
    return container


def _can_group_under(parent: SpanContext | None) -> TypeGuard[SpanContext]:
    """Whether a container under ``parent`` would ever be closed."""
    return parent is not None and parent.is_valid and parent.trace_flags.sampled


def guardrail_aware(
    provider: Callable[[], Span | None],
) -> Callable[[], Span | None]:
    """Report the running check's container as the current span."""

    def current_span() -> Span | None:
        return _current_container.get() or provider()

    return current_span


class GuardrailStageCloser(SpanProcessor):
    """End a container when a non-guardrail sibling starts or its parent ends."""

    def on_start(
        self, span: Span, parent_context: otel_context.Context | None = None
    ) -> None:
        if _starting_container.get() or not _state.containers:
            return
        attributes = getattr(span, "attributes", None) or {}
        parent = getattr(span, "parent", None)
        if attributes.get(_CONTAINER_MARK) or parent is None:
            return
        same_stage = _state.guardrail_nodes.get(getattr(span, "name", ""))
        self._close(
            lambda key: key[0] == parent.span_id and key[1] != same_stage,
            end_time=getattr(span, "start_time", None),
        )

    def on_end(self, span: ReadableSpan) -> None:
        if not _state.containers:
            return
        span_id = span.context.span_id if span.context else None
        self._close(lambda key: key[0] == span_id, end_time=span.end_time)

    def shutdown(self) -> None:
        """Called when its trace manager shuts down: unregister, end what's open."""
        with _state.lock:
            owners = [m for m, closer in _state.closers.items() if closer is self]
            for manager in owners:
                del _state.closers[manager]
            if any(manager is _state.trace_manager for manager in owners):
                _state.trace_manager = None
            _state.enabled = bool(_state.closers)
        if not _state.enabled:
            self._close(lambda key: True, end_time=None)

    @staticmethod
    def _close(
        matches: Callable[[tuple[int, str]], bool], end_time: int | None
    ) -> None:
        with _state.lock:
            done = _pop_containers(matches)
        _end_containers(done, end_time)


def _pop_containers(
    matches: Callable[[tuple[int, str]], bool],
) -> list[tuple[Span, int]]:
    """Remove matching containers, with their last check's end. Hold the lock."""
    keys = [key for key in _state.containers if matches(key)]
    spans = [_state.containers.pop(key) for key in keys]
    return [
        (span, _state.last_check_end.pop(span.get_span_context().span_id, 0))
        for span in spans
    ]


def _end_containers(containers: list[tuple[Span, int]], end_time: int | None) -> None:
    for span, last_end in containers:
        status = getattr(span, "status", None)
        if status is None or status.status_code is StatusCode.UNSET:
            span.set_status(StatusCode.OK)
        # Never before its last check: coarse clocks (Windows before Python
        # 3.13) can stamp the next sibling's start a tick earlier.
        span.end(end_time=max(end_time, last_end) if end_time else None)


def is_exported(span: ReadableSpan) -> bool:
    """Span filter hiding guardrail node spans (by name: spans upload on start)."""
    if not _state.enabled or span.name not in _state.guardrail_nodes:
        return True
    if not _can_group_under(span.parent):
        return True  # its checks stay under it, ungrouped
    scope = getattr(span, "instrumentation_scope", None)
    return "openinference" not in (getattr(scope, "name", "") or "")


def enable_guardrail_grouping(
    trace_manager: Any, current_span_provider: Callable[[], Span | None]
) -> None:
    """Register the span provider and offer the trace manager for grouping.

    Grouping turns on once a guardrail middleware is also in use.
    """
    UiPathSpanUtils.register_current_span_provider(
        guardrail_aware(current_span_provider)
    )
    _state.trace_manager = trace_manager
    _activate()


def _activate() -> None:
    """Attach a closer to the offered trace manager if guardrails are in use."""
    with _state.lock:
        manager = _state.trace_manager
        if manager is None or not _state.in_use:
            return
        if manager not in _state.closers:
            closer = GuardrailStageCloser()
            manager.add_span_processor(closer)
            _state.closers[manager] = closer
        _state.enabled = True
