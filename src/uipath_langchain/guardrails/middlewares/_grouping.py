"""Group a stage's guardrail checks into one container span.

Each guardrail hook runs as its own LangGraph node, so the checks of one stage
(e.g. LLM input) are spread over several nodes. The first check of a stage
starts a container next to those nodes (inside ``tools`` for tool checks);
later checks reuse it. ``GuardrailStageCloser`` ends it when something else
starts beside it or its parent ends, and ``is_exported`` hides the now-empty
guardrail node spans.
"""

import logging
import threading
import time
import weakref
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeGuard

from langgraph.errors import GraphBubbleUp
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext, StatusCode
from uipath.core.guardrails import GuardrailScope
from uipath.core.tracing.span_utils import UiPathSpanUtils
from uipath.platform.guardrails.decorators import GuardrailExecutionStage

from ._span_name import SpanName, SpanType
from ._spans import error_text, new_span

logger = logging.getLogger(__name__)


class _Stage(NamedTuple):
    name: str
    span_type: str


_PRE = GuardrailExecutionStage.PRE
_POST = GuardrailExecutionStage.POST

_STAGES: dict[tuple[GuardrailScope, GuardrailExecutionStage], _Stage] = {
    (GuardrailScope.AGENT, _PRE): _Stage(
        SpanName.AGENT_PRE_GUARDRAILS, SpanType.AGENT_PRE_GUARDRAILS
    ),
    (GuardrailScope.AGENT, _POST): _Stage(
        SpanName.AGENT_POST_GUARDRAILS, SpanType.AGENT_POST_GUARDRAILS
    ),
    (GuardrailScope.LLM, _PRE): _Stage(
        SpanName.LLM_PRE_GUARDRAILS, SpanType.LLM_PRE_GUARDRAILS
    ),
    (GuardrailScope.LLM, _POST): _Stage(
        SpanName.LLM_POST_GUARDRAILS, SpanType.LLM_POST_GUARDRAILS
    ),
    (GuardrailScope.TOOL, _PRE): _Stage(
        SpanName.TOOL_PRE_GUARDRAILS, SpanType.TOOL_PRE_GUARDRAILS
    ),
    (GuardrailScope.TOOL, _POST): _Stage(
        SpanName.TOOL_POST_GUARDRAILS, SpanType.TOOL_POST_GUARDRAILS
    ),
}


class _ContainerKey(NamedTuple):
    parent_id: int
    span_type: str


@dataclass
class _Container:
    span: Span
    last_check_end: int = 0  # ns; the container never ends before this


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
        self.containers: dict[_ContainerKey, _Container] = {}
        self.lock = threading.Lock()


_state = _GroupingState()

_current_container: ContextVar[Span | None] = ContextVar(
    "uipath_guardrail_container", default=None
)
# Set while a container starts, so the closer skips it without taking the lock.
_starting_container: ContextVar[bool] = ContextVar(
    "uipath_guardrail_container_starting", default=False
)


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


def guardrails_in_use() -> None:
    """Called by guardrail middlewares when built; grouping needs one in use."""
    _state.in_use = True
    _activate()


def register_guardrail_node(
    node_name: str, scope: GuardrailScope, stage: GuardrailExecutionStage
) -> None:
    """Record which stage a guardrail hook's LangGraph node belongs to."""
    _state.guardrail_nodes[node_name] = _STAGES[(scope, stage)].span_type
    guardrails_in_use()


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
    token = _current_container.set(container.span)
    try:
        yield
    except GraphBubbleUp:
        raise
    except Exception as exc:
        container.span.set_status(StatusCode.ERROR, error_text(exc))
        raise
    finally:
        _current_container.reset(token)
        container.last_check_end = time.time_ns()


def guardrail_aware(
    provider: Callable[[], Span | None],
) -> Callable[[], Span | None]:
    """Report the running check's container as the current span."""

    def current_span() -> Span | None:
        return _current_container.get() or provider()

    return current_span


def _shared_container(
    scope: GuardrailScope | None, stage: GuardrailExecutionStage | None
) -> _Container | None:
    """Get the stage's open container, or start one."""
    if not _state.enabled or (scope, stage) not in _STAGES:
        return None
    node = UiPathSpanUtils.get_external_current_span()
    if node is None or not node.is_recording():
        return None  # a sampled-out node never reports its end
    parent = _container_parent(node, scope)
    if not _can_group_under(parent):
        return None

    stage_spans = _STAGES[(scope, stage)]  # type: ignore[index]
    key = _ContainerKey(parent.span_id, stage_spans.span_type)
    with _state.lock:
        container = _state.containers.get(key)
        if container is not None:
            return container
        container = _Container(_start_container(stage_spans, parent, node))
        # The other stage under this parent is over, even if no sibling
        # started in between (a middleware that answers a tool call itself).
        other_stage = _pop_containers(
            lambda k: k.parent_id == key.parent_id and k.span_type != key.span_type
        )
        _state.containers[key] = container
    _end_containers(other_stage, end_time=getattr(container.span, "start_time", None))
    return container


def _container_parent(node: Span, scope: GuardrailScope | None) -> SpanContext | None:
    """Tool checks go inside the ``tools`` node; others beside their own node."""
    if scope == GuardrailScope.TOOL:
        return node.get_span_context()
    return getattr(node, "parent", None)


def _can_group_under(parent: SpanContext | None) -> TypeGuard[SpanContext]:
    """Whether a container under ``parent`` would ever be closed."""
    return parent is not None and parent.is_valid and parent.trace_flags.sampled


def _start_container(stage: _Stage, parent: SpanContext, node: Span) -> Span:
    token = _starting_container.set(True)
    try:
        return new_span(
            stage.name,
            stage.span_type,
            context=trace.set_span_in_context(NonRecordingSpan(parent)),
            **{
                # Its parent is non-recording, so this isn't inherited.
                "execution.id": (getattr(node, "attributes", None) or {}).get(
                    "execution.id"
                ),
            },
        )
    finally:
        _starting_container.reset(token)


class GuardrailStageCloser(SpanProcessor):
    """End a container when a non-guardrail sibling starts or its parent ends."""

    def on_start(
        self, span: Span, parent_context: otel_context.Context | None = None
    ) -> None:
        if _starting_container.get() or not _state.containers:
            return
        parent = getattr(span, "parent", None)
        if parent is None:
            return
        # A guardrail node of the same stage joins the container; anything
        # else starting beside it ends the stage.
        same_stage = _state.guardrail_nodes.get(getattr(span, "name", ""))
        _close(
            lambda key: key.parent_id == parent.span_id and key.span_type != same_stage,
            end_time=getattr(span, "start_time", None),
        )

    def on_end(self, span: ReadableSpan) -> None:
        if not _state.containers:
            return
        span_id = span.context.span_id if span.context else None
        _close(lambda key: key.parent_id == span_id, end_time=span.end_time)

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
            _close(lambda key: True, end_time=None)


def _close(matches: Callable[[_ContainerKey], bool], end_time: int | None) -> None:
    with _state.lock:
        done = _pop_containers(matches)
    _end_containers(done, end_time)


def _pop_containers(matches: Callable[[_ContainerKey], bool]) -> list[_Container]:
    """Remove the matching open containers. Call with the lock held."""
    keys = [key for key in _state.containers if matches(key)]
    return [_state.containers.pop(key) for key in keys]


def _end_containers(containers: list[_Container], end_time: int | None) -> None:
    for container in containers:
        span = container.span
        status = getattr(span, "status", None)
        if status is None or status.status_code is StatusCode.UNSET:
            span.set_status(StatusCode.OK)
        # Never before its last check: coarse clocks (Windows before Python
        # 3.13) can stamp the next sibling's start a tick earlier.
        span.end(end_time=max(end_time, container.last_check_end) if end_time else None)


def is_exported(span: ReadableSpan) -> bool:
    """Hide guardrail node spans. Decided by name: a span uploads when it starts."""
    if not _state.enabled or span.name not in _state.guardrail_nodes:
        return True
    if not _can_group_under(span.parent):
        return True  # its checks stay under it, ungrouped
    scope = getattr(span, "instrumentation_scope", None)
    return "openinference" not in (getattr(scope, "name", "") or "")
