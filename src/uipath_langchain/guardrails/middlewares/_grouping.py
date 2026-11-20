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


@dataclass
class _Container:
    span: Span
    span_type: str
    last_check_end: int = 0  # ns

    def end(self, end_time: int | None) -> None:
        status = getattr(self.span, "status", None)
        if status is None or status.status_code is StatusCode.UNSET:
            self.span.set_status(StatusCode.OK)
        # Never before its last check: coarse clocks (Windows before Python
        # 3.13) can stamp the next sibling's start a tick earlier.
        self.span.end(end_time=max(end_time, self.last_check_end) if end_time else None)


class _GroupingState:
    """Process-wide state. Grouping is on while a closer is attached."""

    def __init__(self) -> None:
        self.in_use = False  # a guardrail middleware was built
        self.trace_manager: Any = None  # offered by the runtime factory
        self.closer: GuardrailStageCloser | None = None
        self.guardrail_nodes: dict[str, str] = {}  # node name -> stage span type
        # Parent span id -> its open container. One at a time: starting a
        # stage's container ends the other stage's one under the same parent.
        self.containers: dict[int, _Container] = {}
        # Re-entrant: starting a container runs the closer's on_start, which
        # takes it again on the same thread.
        self.lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.closer is not None


_state = _GroupingState()

_current_container: ContextVar[Span | None] = ContextVar(
    "uipath_guardrail_container", default=None
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
        if _state.closer is None or _state.closer.trace_manager is not manager:
            _state.closer = GuardrailStageCloser(manager)
            manager.add_span_processor(_state.closer)


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
    """Get the stage's open container, or start one. ``None`` if not grouped."""
    stage_spans = _STAGES.get((scope, stage))  # type: ignore[arg-type]
    if not _state.enabled or stage_spans is None:
        return None
    place = _container_place(scope)
    if place is None:
        return None
    node, parent = place
    return _open_container(stage_spans, parent, node)


def _container_place(scope: GuardrailScope | None) -> tuple[Span, SpanContext] | None:
    """The check's node and the container's parent, if a container there would close.

    Tool checks go inside the ``tools`` node; others beside their own node.
    """
    node = UiPathSpanUtils.get_external_current_span()
    if node is None or not node.is_recording():
        return None  # a sampled-out node never reports its end
    if scope == GuardrailScope.TOOL:
        parent: SpanContext | None = node.get_span_context()
    else:
        parent = getattr(node, "parent", None)
    if not _can_group_under(parent):
        return None
    return node, parent


def _open_container(stage: _Stage, parent: SpanContext, node: Span) -> _Container:
    """Reuse ``stage``'s open container under ``parent``, or start it there.

    Starting it ends the other stage's container under the same parent, even if
    no sibling started in between (a middleware that answers a tool call itself).
    """
    with _state.lock:
        previous = _state.containers.get(parent.span_id)
        if previous is not None and previous.span_type == stage.span_type:
            return previous
        # Out while the new one starts, so the closer doesn't end it under the
        # lock; back in if the start fails.
        _state.containers.pop(parent.span_id, None)
        try:
            container = _start_container(stage, parent, node)
        except Exception:
            if previous is not None:
                _state.containers[parent.span_id] = previous
            raise
        _state.containers[parent.span_id] = container
    if previous is not None:
        previous.end(getattr(container.span, "start_time", None))
    return container


def _can_group_under(parent: SpanContext | None) -> TypeGuard[SpanContext]:
    """Whether a container under ``parent`` would ever be closed."""
    return parent is not None and parent.is_valid and parent.trace_flags.sampled


def _start_container(stage: _Stage, parent: SpanContext, node: Span) -> _Container:
    """Start ``stage``'s container under ``parent``."""
    span = new_span(
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
    return _Container(span, stage.span_type)


class GuardrailStageCloser(SpanProcessor):
    """End a container when a non-guardrail sibling starts or its parent ends."""

    def __init__(self, trace_manager: Any = None) -> None:
        self.trace_manager = trace_manager

    def on_start(
        self, span: ReadableSpan, parent_context: otel_context.Context | None = None
    ) -> None:
        if not _state.containers or span.parent is None:
            return
        with _state.lock:
            container = _state.containers.get(span.parent.span_id)
            # A guardrail node of the same stage joins the container; anything
            # else starting beside it ends the stage.
            if container is None or container.span_type == _state.guardrail_nodes.get(
                span.name
            ):
                return
            del _state.containers[span.parent.span_id]
        container.end(span.start_time)

    def on_end(self, span: ReadableSpan) -> None:
        if not _state.containers or span.context is None:
            return
        with _state.lock:
            container = _state.containers.pop(span.context.span_id, None)
        if container is not None:
            container.end(span.end_time)

    def shutdown(self) -> None:
        """Called when its trace manager shuts down: detach, end what's open."""
        with _state.lock:
            if self is not _state.closer:
                return
            _state.closer = None
            if _state.trace_manager is self.trace_manager:
                _state.trace_manager = None
            still_open = list(_state.containers.values())
            _state.containers.clear()
        for container in still_open:
            container.end(None)


def is_exported(span: ReadableSpan) -> bool:
    """Hide guardrail node spans. Decided by name: a span uploads when it starts."""
    if not _state.enabled or span.name not in _state.guardrail_nodes:
        return True
    if not _can_group_under(span.parent):
        return True  # its checks stay under it, ungrouped
    scope = getattr(span, "instrumentation_scope", None)
    return "openinference" not in (getattr(scope, "name", "") or "")
