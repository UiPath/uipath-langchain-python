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

import logging
from collections.abc import Iterable, Sequence
from typing import Any, Literal, get_args

from deepagents import CompiledSubAgent, FsToolName, SubAgent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool

from uipath_langchain._utils.durable_interrupt import suspends_run

logger = logging.getLogger(__name__)

_MISSING_EXTRA = (
    "The code interpreter needs the 'code-interpreter' extra. Install it with "
    '`uv add "uipath-langchain[code-interpreter]"` (or `pip install '
    '"uipath-langchain[code-interpreter]"`).'
)

# ``FilesystemMiddleware`` adds these after we are handed the tool list, so their
# names have to be supplied rather than read off ``tools``. Upstream matches a
# ``ptc`` name against the live tool list and ignores one that is absent, so
# listing the whole literal exposes exactly the tools the backend supports:
# ``execute`` only for a ``SandboxBackendProtocol`` backend, which ours is not.
PTC_FILESYSTEM_TOOLS: tuple[str, ...] = get_args(FsToolName)

_RESERVED_TOOL_NAMES = frozenset({"task"})

# Subagent spec keys that make deepagents interrupt without a stamped tool.
_SUBAGENT_INTERRUPT_KEYS = ("interrupt_on", "permissions", "middleware")

PersistenceMode = Literal["thread", "turn", "call"]
"""How long the REPL keeps state. Mirrors ``langchain_quickjs.PersistenceMode``
rather than importing ``langchain_quickjs.middleware.PersistenceMode``, so this
module imports without the optional extra.

``"thread"`` writes a snapshot of the interpreter's memory into the checkpoint on
every run, measured at ~1.25 MB even when the agent never calls ``eval``. The
other two write nothing.
"""

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
    - **Names that cannot be JavaScript identifiers.** A tool name is caller
      supplied and may hold spaces, dots or non-ASCII characters. Upstream raises
      ``ValueError`` for those from inside ``wrap_model_call``, faulting the run
      mid-turn, so they are dropped here instead.
    - **camelCase collisions.** ``get_invoice`` and ``get-invoice`` both become
      ``getInvoice``, and upstream dedupes by tool name rather than camel name
      while binding by camel name last-wins, so one of the two silently answers
      for both and which one depends on tool order. Every member of a colliding
      group is dropped, including a group formed against
      :data:`PTC_FILESYSTEM_TOOLS`: a tool named ``read-file`` would otherwise
      take over the ``tools.readFile`` the REPL prompt documents as the
      workspace reader.

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

    return [
        t.name
        for t in _without_camel_collisions(
            eligible, to_camel, reserved={to_camel(n) for n in PTC_FILESYSTEM_TOOLS}
        )
    ]


def subagent_dispatch_is_replay_safe(
    subagents: Sequence[SubAgent | CompiledSubAgent],
    shared_tools: Sequence[BaseTool],
) -> bool:
    """Whether ``task()`` can be offered inside the REPL.

    An interrupt raised while an ``eval`` is still running replays the whole
    ``eval`` on resume, re-running every bridged call it already made. Excluding
    suspending tools from ``ptc`` closes the direct route, but ``task()`` reaches a
    subagent's tools through a path that allowlist does not cover, so a subagent
    that can suspend reopens it.

    Withholding ``task()`` costs single-turn orchestration, not subagent dispatch:
    ``task`` stays an ordinary tool, where the interrupt checkpoints correctly.

    A subagent inherits ``shared_tools`` unless its spec declares ``tools``, and
    deepagents adds a general-purpose subagent that inherits them too, so a
    suspending tool on the main agent withholds dispatch on its own. A
    ``CompiledSubAgent`` brings a graph whose tools cannot be read, so it counts
    against dispatch rather than being assumed safe.

    :data:`SUSPENDS_RUN` only marks our own tools, so it does not see the HITL
    deepagents builds from a spec: ``interrupt_on`` becomes a
    ``HumanInTheLoopMiddleware``, ``permissions`` folds into ``interrupt_on``, and
    ``middleware`` can carry one directly. Each of those interrupts with no
    stamped tool involved, so declaring any of them withholds dispatch too.
    """
    if any(suspends_run(tool) for tool in shared_tools):
        return False
    for spec in subagents:
        name = spec.get("name", "<unnamed>")
        if "runnable" in spec:
            logger.info(
                "task() withheld from the REPL: subagent %r is precompiled, so its "
                "tools cannot be checked for run suspension",
                name,
            )
            return False
        if any(spec.get(key) for key in _SUBAGENT_INTERRUPT_KEYS):
            logger.info(
                "task() withheld from the REPL: subagent %r declares its own "
                "human-in-the-loop configuration",
                name,
            )
            return False
        if any(suspends_run(tool) for tool in spec.get("tools", ())):
            logger.info(
                "task() withheld from the REPL: subagent %r holds a tool that "
                "suspends the run",
                name,
            )
            return False
    return True


def build_code_interpreter_middleware(
    tools: Sequence[BaseTool],
    *,
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    mode: PersistenceMode = "thread",
    timeout: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> list[AgentMiddleware[Any, Any]]:
    """The code-interpreter middleware for ``tools``, ready to pass as ``middleware``.

    Returned as a list so a caller can splice it into a middleware sequence
    without branching.

    Args:
        tools: The agent's tools. Eligible ones become callable from the REPL.
        mode: How long the REPL keeps state; see :data:`PersistenceMode`. A
            caller whose runs do not share a checkpoint thread should pass
            ``"turn"``, since ``"thread"`` would pay the snapshot cost per run
            and never read it back.
        subagents: The agent's subagent specs. Whether ``task()`` is offered
            inside the REPL is derived from them; see
            :func:`subagent_dispatch_is_replay_safe`.
        timeout: Per-eval wall clock in seconds.

    Raises:
        ImportError: If the ``code-interpreter`` extra is not installed.
    """
    middleware_cls = _code_interpreter_middleware_cls()
    exposed = ptc_tool_names(tools)
    dispatch = subagent_dispatch_is_replay_safe(subagents, tools)
    logger.info(
        "Code interpreter enabled: %d of %d agent tools exposed for PTC, "
        "task() %s in the REPL",
        len(exposed),
        len(tools),
        "offered" if dispatch else "withheld",
    )
    return [
        middleware_cls(
            ptc=[*exposed, *PTC_FILESYSTEM_TOOLS],
            mode=mode,
            subagents=dispatch,
            timeout=timeout,
        )
    ]


def _without_camel_collisions(
    tools: Iterable[BaseTool], to_camel: Any, reserved: set[str]
) -> list[BaseTool]:
    """Drop every tool whose camelCase name is not uniquely its own."""
    by_camel: dict[str, list[BaseTool]] = {}
    for tool in tools:
        by_camel.setdefault(to_camel(tool.name), []).append(tool)

    kept: list[BaseTool] = []
    for camel, group in by_camel.items():
        if camel in reserved:
            logger.warning(
                "Tools %s withheld from PTC: %r is a workspace tool",
                [t.name for t in group],
                camel,
            )
            continue
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
