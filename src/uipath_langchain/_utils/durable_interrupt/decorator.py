"""Durable interrupt: side-effect-safe interrupt/resume for LangGraph subgraphs.

LangGraph's ``@task`` + ``interrupt()`` pattern ensures that a side-effecting
operation (starting a process, creating an escalation task, etc.) runs exactly
once even when the graph resumes from the interrupt.  However, ``@task``'s
checkpoint-based caching breaks when the tool node is wrapped in a **subgraph**
(e.g. guardrails), because the subgraph's Pregel context does not reliably
preserve per-task checkpoints across interrupt/resume boundaries.

``@durable_interrupt`` replaces the ``@task`` + ``interrupt()`` two-step
pattern with a single decorator that guarantees exactly-once execution and
correct interrupt/resume semantics::

    @durable_interrupt
    async def start_job():
        client = UiPath()
        job = await client.processes.invoke_async(...)
        return WaitJob(job=job, ...)

    # First run:  body executes → interrupt(WaitJob(...)) → GraphInterrupt
    # Resume:     body skipped  → interrupt(None) → returns resume value
    result = await start_job()

The decorated function's return value is passed directly to ``interrupt()``.
On resume, the body is skipped and ``interrupt(None)`` returns the resume
value from the runtime.

Multiple ``@durable_interrupt`` calls in the same node are supported.  An
internal per-node counter (auto-reset via scratchpad identity) keeps our
index in sync with LangGraph's own ``interrupt_counter``.
"""

import asyncio
import contextvars
import functools
from typing import Any, Callable, TypeVar

from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph.config import get_config
from langgraph.types import interrupt

from .skip_interrupt import SkipInterruptValue

F = TypeVar("F", bound=Callable[..., Any])

# Tracks (scratchpad identity, call index) per node execution.
# Resets automatically when the scratchpad changes (new node execution).
_durable_state: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar(
    "_durable_interrupt_state", default=None
)


def _next_durable_index() -> tuple[Any, int]:
    """Return (scratchpad, index) for the current durable_interrupt call.

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


def _inject_resume(scratchpad: Any, value: Any) -> Any:
    """Inject a value into the scratchpad resume list and return it via interrupt(None).

    This keeps LangGraph's interrupt_counter in sync (interrupt(None) increments it)
    while avoiding a real suspend — interrupt(None) finds the injected value and
    returns it immediately without raising GraphInterrupt.
    """
    if scratchpad is not None:
        if scratchpad.resume is None:
            scratchpad.resume = []
        scratchpad.resume.append(value)
        return interrupt(None)
    return value


SUSPENDS_RUN = "suspends_run"
"""Tool-metadata key marking a tool that can suspend the run instead of returning.

Such a tool may raise ``GraphInterrupt`` rather than produce a value: the run
checkpoints, and the node is replayed from that checkpoint on resume. Callers that
invoke tools *outside* the graph's tool node -- the QuickJS code interpreter's
programmatic tool calling, for one -- must not offer these, because a replayed
node re-runs every call made before the interrupt, and because such bridges
bypass approval hooks.

The flag describes that behaviour rather than how it is reached. It covers both
``durable_interrupt`` and a bare ``interrupt()``, and it is set unconditionally on
a tool that suspends only sometimes -- one returning a ``SkipInterruptValue`` on
its fast path, say -- since the safe answer for a caller outside the tool node is
the same either way.

Set it in the tool's ``metadata`` in every factory that can suspend;
``tests/agent/tools/test_suspends_run_metadata.py`` fails if one forgets.
"""


def suspends_run(tool: Any) -> bool:
    """Whether ``tool`` suspends the run instead of returning a value.

    Reads :data:`SUSPENDS_RUN` from the tool's metadata, so it is accurate
    per-tool even where one factory builds both suspending and non-suspending
    variants (``context_tool`` does, by retrieval mode).

    Fails closed only where the flag is present: an unstamped tool reports
    ``False``. ``tests/agent/tools/test_suspends_run_metadata.py`` is what keeps
    that safe, by failing when a suspending factory ships without the stamp.
    """
    return bool((getattr(tool, "metadata", None) or {}).get(SUSPENDS_RUN))


def durable_interrupt(fn: F) -> F:
    """Decorator that executes a side-effecting function exactly once and interrupts.

    On first execution the body runs and its return value is passed to
    ``interrupt()`` (which raises ``GraphInterrupt``).  On resume the body
    is skipped and ``interrupt(None)`` returns the resume value from the
    runtime.

    If the body returns a ``SkipInterruptValue``, the resolved value is
    injected into the scratchpad resume list and ``interrupt(None)`` returns
    it immediately — no real suspend/resume cycle occurs.

    Replaces the ``@task`` + ``interrupt()`` two-step pattern with a single
    decorator that enforces the pairing contract.  Works correctly in both
    parent graphs and subgraphs.

    Supports both sync and async functions::

        @durable_interrupt
        async def create_task():
            return await client.tasks.create_async(...)

        # Returns resume value on resume, raises GraphInterrupt on first run
        result = await create_task()

        @durable_interrupt
        def create_task_sync():
            return client.tasks.create(...)

        result = create_task_sync()
    """

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            scratchpad, idx = _next_durable_index()
            if _is_resumed(scratchpad, idx):
                return interrupt(None)
            result = await fn(*args, **kwargs)
            if isinstance(result, SkipInterruptValue):
                return _inject_resume(scratchpad, result.resume_value)
            return interrupt(result)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        scratchpad, idx = _next_durable_index()
        if _is_resumed(scratchpad, idx):
            return interrupt(None)
        result = fn(*args, **kwargs)
        if isinstance(result, SkipInterruptValue):
            return _inject_resume(scratchpad, result.resume_value)
        return interrupt(result)

    return sync_wrapper  # type: ignore[return-value]
