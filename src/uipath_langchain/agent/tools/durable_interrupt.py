"""Durable interrupt: side-effect-safe interrupt/resume for LangGraph subgraphs.

LangGraph's ``@task`` + ``interrupt()`` pattern ensures that a side-effecting
operation (starting a process, creating an escalation task, etc.) runs exactly
once even when the graph resumes from the interrupt.  However, ``@task``'s
checkpoint-based caching breaks when the tool node is wrapped in a **subgraph**
(e.g. guardrails), because the subgraph's Pregel context does not reliably
preserve per-task checkpoints across interrupt/resume boundaries.

``durable_interrupt`` replaces that pattern: it inspects the LangGraph
scratchpad to decide whether the current interrupt index has already been
resolved.  If so, it skips the action and lets ``interrupt()`` return the
cached resume value.  If not, it executes the action and interrupts.

Multiple ``durable_interrupt`` calls in the same node are supported.  An
internal per-node counter (auto-reset via scratchpad identity) keeps our
index in sync with LangGraph's own ``interrupt_counter``.
"""

import contextvars
from typing import Any, Awaitable, Callable, TypeVar

from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph.config import get_config
from langgraph.types import interrupt

T = TypeVar("T")

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


async def durable_interrupt(
    action: Callable[[], Awaitable[T]],
    make_interrupt_value: Callable[[T], Any],
) -> Any:
    """Execute action once and interrupt; on resume, skip the action and return the resume value.

    Replaces the @task + interrupt() pattern.  Works correctly in both
    parent graphs and subgraphs (e.g. guardrails), where @task's
    checkpoint-based caching could cause double invocations.

    Safe with multiple calls in the same node: each call tracks its own
    interrupt index and only skips the action when a resume value exists
    for that specific index.

    Args:
        action: Async callable that performs the side-effecting operation.
        make_interrupt_value: Builds the interrupt payload from the action result.

    Returns:
        The resume value provided by the runtime when the graph continues.
    """
    scratchpad, idx = _next_durable_index()
    already_resumed = (
        scratchpad is not None and scratchpad.resume and idx < len(scratchpad.resume)
    )

    if already_resumed:
        return interrupt(None)
    else:
        result = await action()
        return interrupt(make_interrupt_value(result))
