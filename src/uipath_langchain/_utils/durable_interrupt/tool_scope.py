"""Which tool's own node is running, for tools that suspend only there.

A tool that may raise ``GraphInterrupt`` can do so safely only when the graph
node being run is *its* node: that node is replayed on resume, and the tool's
durable steps make the replay harmless. Reached any other way -- from the code
interpreter's bridge, which runs inside the ``eval`` tool's node, for one --
an interrupt would replay somebody else's node and every call it had made.

A tool that always suspends is simply withheld from such callers
(:data:`SUSPENDS_RUN`). A tool that suspends only sometimes, and has a way to
finish without suspending, can stay available to them and ask here instead.
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

_running_tool: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "_tool_node_running_tool", default=None
)


@contextmanager
def tool_node_scope(tool: Any) -> Iterator[None]:
    """Mark ``tool`` as the tool whose node is running, for the enclosed call."""
    token = _running_tool.set(tool)
    try:
        yield
    finally:
        _running_tool.reset(token)


def runs_in_own_tool_node(tool: Any) -> bool:
    """Whether the node being run is ``tool``'s own tool node."""
    return tool is not None and _running_tool.get() is tool
