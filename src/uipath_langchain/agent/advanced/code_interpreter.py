"""The QuickJS code interpreter for advanced agents, and what it may call.

``CodeInterpreterMiddleware`` adds one ``eval`` tool: a persistent JavaScript REPL
in a WASM guest (QuickJS-ng under wasmtime). It serves three purposes in a single
tool call -- computation, programmatic tool calling (PTC), and subagent
orchestration through the top-level ``task()`` global.

The guest has no ambient capability: no network, no filesystem, no ``fetch``, no
``require``, no timers. Everything it can reach arrives through the ``ptc``
allowlist, which makes that allowlist the entire security surface of the feature.
It is derived here rather than configured, because the rule that governs it is a
property of our tools (see :data:`SUSPENDS_RUN`) and not of any one consumer.

Requires the ``code-interpreter`` extra::

    uv add "uipath-langchain[code-interpreter]"
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool

from uipath_langchain._utils.durable_interrupt import suspends_run

logger = logging.getLogger(__name__)

_MISSING_EXTRA = (
    "The code interpreter needs the 'code-interpreter' extra. Install it with "
    '`uv add "uipath-langchain[code-interpreter]"` (or `pip install '
    '"uipath-langchain[code-interpreter]"`).'
)

# deepagents' own file tools, safe to reach from inside the REPL: each returns a
# value, and each routes through the backend, so ``virtual_mode`` still resolves
# and bounds every path. ``delete`` and ``execute`` are deliberately absent, and
# ``task`` is reserved -- upstream raises if it is listed, because it is exposed
# as the ``task()`` global instead.
PTC_FILESYSTEM_TOOLS: tuple[str, ...] = (
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
)

_RESERVED_TOOL_NAMES = frozenset({"task"})

# Per-eval wall clock. The REPL is for orchestration and arithmetic, not long
# computation, and a bridged tool call does not consume it.
DEFAULT_EVAL_TIMEOUT_SECONDS = 5.0


def ptc_tool_names(tools: Sequence[BaseTool]) -> list[str]:
    """Names of the agent tools that may be called from inside the REPL.

    Three exclusions, each for a different reason:

    - **Tools that suspend the run.** One raising ``GraphInterrupt`` never returns
      a value into the JS ``await``. Worse, the node is replayed from its
      checkpoint on resume, so the ``eval`` re-runs from the top and every bridged
      call made before the interrupt fires a second time. Upstream also documents
      that PTC bridges bypass ``interrupt_on`` approval hooks, so an escalation
      reached this way would skip its own approval.
    - **Names that cannot be JavaScript identifiers.** Low-code tool names come
      from ``agent.json`` and may hold spaces, dots or non-ASCII characters.
      Upstream raises ``ValueError`` for those from inside ``wrap_model_call``,
      faulting the run mid-turn, so they are dropped here instead.
    - **camelCase collisions.** ``get_invoice`` and ``get-invoice`` both become
      ``getInvoice``, and upstream dedupes by tool name rather than camel name.
      Every member of a colliding group is dropped: binding one of two
      identically-named JS functions would silently call the wrong tool.

    An excluded tool stays fully available as an ordinary tool call, so exclusion
    costs a model round trip, never a capability.
    """
    is_valid, to_camel = _name_validators()

    eligible: list[BaseTool] = []
    for tool in tools:
        if tool.name in _RESERVED_TOOL_NAMES:
            continue
        if suspends_run(tool):
            logger.debug("Tool %r withheld from PTC: it suspends the run", tool.name)
            continue
        if not is_valid(tool.name):
            logger.info(
                "Tool %r withheld from PTC: %r is not a valid JavaScript identifier",
                tool.name,
                to_camel(tool.name),
            )
            continue
        eligible.append(tool)

    return [t.name for t in _without_camel_collisions(eligible, to_camel)]


def build_code_interpreter_middleware(
    tools: Sequence[BaseTool],
    *,
    timeout: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
    subagents: bool = True,
) -> list[AgentMiddleware[Any, Any]]:
    """The code-interpreter middleware for ``tools``, ready to pass as ``middleware``.

    Returned as a list so a caller can splice it into a middleware sequence
    without branching.

    Args:
        tools: The agent's tools. Eligible ones become callable from the REPL.
        timeout: Per-eval wall clock in seconds.
        subagents: Expose the top-level ``task()`` global when the host has a
            deepagents ``task`` tool. A no-op for an agent with no subagents.

    Raises:
        ImportError: If the ``code-interpreter`` extra is not installed.
    """
    middleware_cls = _code_interpreter_middleware_cls()
    exposed = ptc_tool_names(tools)
    logger.info(
        "Code interpreter enabled: %d of %d agent tools exposed for PTC",
        len(exposed),
        len(tools),
    )
    return [
        middleware_cls(
            ptc=[*exposed, *PTC_FILESYSTEM_TOOLS],
            # Globals persist across turns of a LangGraph thread. Whether that
            # survives a suspend/resume boundary is unverified; "turn" is the
            # fallback if it does not.
            mode="thread",
            subagents=subagents,
            timeout=timeout,
        )
    ]


def _without_camel_collisions(
    tools: Iterable[BaseTool], to_camel: Any
) -> list[BaseTool]:
    """Drop every tool whose camelCase name is shared with another tool."""
    by_camel: dict[str, list[BaseTool]] = {}
    for tool in tools:
        by_camel.setdefault(to_camel(tool.name), []).append(tool)

    kept: list[BaseTool] = []
    for camel, group in by_camel.items():
        if len(group) > 1:
            logger.warning(
                "Tools %s withheld from PTC: their names all map to %r",
                [t.name for t in group],
                camel,
            )
            continue
        kept.append(group[0])
    return kept


def _code_interpreter_middleware_cls() -> Any:
    """Import ``CodeInterpreterMiddleware``, or raise with install guidance."""
    try:
        from langchain_quickjs import CodeInterpreterMiddleware
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MISSING_EXTRA) from exc
    return CodeInterpreterMiddleware


def _name_validators() -> tuple[Any, Any]:
    """Upstream's identifier rule and camelCase conversion.

    Taken from ``langchain_quickjs._ptc`` rather than reimplemented: a local copy
    risks drifting *looser* than upstream, and anything upstream rejects raises
    from inside ``wrap_model_call``, faulting the run rather than degrading. The
    import is pinned by ``tests/agent/advanced/test_code_interpreter.py``.
    """
    try:
        from langchain_quickjs._ptc import is_valid_ptc_tool_name, to_camel_case
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MISSING_EXTRA) from exc
    return is_valid_ptc_tool_name, to_camel_case
