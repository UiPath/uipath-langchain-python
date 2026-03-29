"""Skip-interrupt value types for @durable_interrupt.

SkipInterruptValue — base class
================================

When a node has **multiple sequential @durable_interrupt calls**, the
decorator's internal counter must produce the same sequence of indices on
every execution (first run *and* all resume runs).  A conditional interrupt
— one that only fires under certain conditions — breaks this assumption and
causes index drift on resume.

``SkipInterruptValue`` solves this by letting a @durable_interrupt-decorated
function signal "the result is already available, skip the real interrupt"
while still keeping LangGraph's interrupt counter in sync.  The decorator
injects the resolved value into ``scratchpad.resume`` and calls
``interrupt(None)``, which returns immediately without raising ``GraphInterrupt``.

Usage example::

    @durable_interrupt
    async def create_index():
        index = await client.create_index_async(...)
        if index.in_progress():
            return WaitIndex(index=index)        # real interrupt
        return ReadyIndex(index=index)            # instant resume

    @durable_interrupt
    async def start_processing():
        return StartProcessing(index_id=index.id) # real interrupt

    # Both @durable_interrupt calls always execute — the counter always
    # increments by 2.  When the index is ready, ReadyIndex (a
    # SkipInterruptValue subclass) injects the result into the scratchpad
    # so the graph continues without suspending.
"""

from typing import Any


class SkipInterruptValue:
    """Base class for values that skip the interrupt in @durable_interrupt.

    Subclasses must implement the ``resume_value`` property, returning the
    value to inject into the scratchpad resume list.
    """

    @property
    def resume_value(self) -> Any:
        """The value to inject into the resume list and return to the caller."""
        raise NotImplementedError
