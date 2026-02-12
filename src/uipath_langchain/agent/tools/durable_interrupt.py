"""Durable task: side-effect-safe interrupt/resume for LangGraph subgraphs.

LangGraph's ``@task`` + ``interrupt()`` pattern ensures that a side-effecting
operation (starting a process, creating an escalation task, etc.) runs exactly
once even when the graph resumes from the interrupt.  However, ``@task``'s
checkpoint-based caching breaks when the tool node is wrapped in a **subgraph**
(e.g. guardrails), because the subgraph's Pregel context does not reliably
preserve per-task checkpoints across interrupt/resume boundaries.

``@durable_task`` is a drop-in replacement for ``@task``.  Decorate a sync or
async function that performs a side-effecting operation.  On first execution the
body runs normally and returns its result.  On resume the body is skipped
and ``None`` is returned (the caller's ``interrupt()`` call will consume
the resume value regardless of its argument).

Multiple ``@durable_task`` calls in the same node are supported.  An internal
per-node counter (auto-reset via scratchpad identity) keeps our index in sync
with LangGraph's own ``interrupt_counter``.
"""

import asyncio
import contextvars
import functools
from typing import Any, Callable, TypeVar

from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph.config import get_config

F = TypeVar("F", bound=Callable[..., Any])

# Tracks (scratchpad identity, call index) per node execution.
# Resets automatically when the scratchpad changes (new node execution).
_durable_state: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar(
    "_durable_interrupt_state", default=None
)


def _next_durable_index() -> tuple[Any, int]:
    """Return (scratchpad, index) for the current durable_task call.

    The index auto-resets when the scratchpad object changes, which happens
    at the start of each node execution.  This keeps our counter in sync with
    LangGraph's internal ``interrupt_counter`` without touching it directly.
    """
    try:
        conf = get_config()
        scratchpad = (conf.get("configurable") or {}).get(CONFIG_KEY_SCRATCHPAD)
    except RuntimeError:
        scratchpad = None

    sp_id = id(scratchpad) if scratchpad else 0
    state = _durable_state.get()

    if state is None or state[0] != sp_id:
        idx = 0
    else:
        idx = state[1]

    _durable_state.set((sp_id, idx + 1))
    return scratchpad, idx


def _is_resumed(scratchpad: Any, idx: int) -> bool:
    return scratchpad is not None and scratchpad.resume and idx < len(scratchpad.resume)


def durable_task(fn: F) -> F:
    """Decorator that runs the function body exactly once across interrupt/resume cycles.

    On first execution the body runs normally and returns its result.
    On resume the body is skipped and ``None`` is returned.

    Drop-in replacement for ``@task`` that works correctly in both parent
    graphs and subgraphs (e.g. guardrails).  The caller is responsible for
    calling ``interrupt()`` with the return value.

    Supports both sync and async functions::

        @durable_task
        async def create_task():
            return await client.tasks.create_async(...)

        created_task = await create_task()
        return interrupt(WaitEscalation(action=created_task, ...))

        @durable_task
        def create_task_sync():
            return client.tasks.create(...)

        created_task = create_task_sync()
        return interrupt(WaitEscalation(action=created_task, ...))
    """

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            scratchpad, idx = _next_durable_index()
            if _is_resumed(scratchpad, idx):
                return None
            return await fn(*args, **kwargs)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        scratchpad, idx = _next_durable_index()
        if _is_resumed(scratchpad, idx):
            return None
        return fn(*args, **kwargs)

    return sync_wrapper  # type: ignore[return-value]
