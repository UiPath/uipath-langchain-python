"""Python version compatibility shims applied at import time."""

import asyncio.base_events
import sys

if sys.version_info < (3, 12):
    # Backport CPython 3.12 fix: BaseEventLoop.is_closed() uses getattr(self, '_closed', True)
    # so that BaseEventLoop.__del__ doesn't raise AttributeError when GC collects a
    # partially-initialized loop (Homebrew Python 3.11 bug, fixed in CPython 3.12).
    def _safe_is_closed(self: asyncio.base_events.BaseEventLoop) -> bool:
        return getattr(self, "_closed", True)

    asyncio.base_events.BaseEventLoop.is_closed = _safe_is_closed  # type: ignore[method-assign]
