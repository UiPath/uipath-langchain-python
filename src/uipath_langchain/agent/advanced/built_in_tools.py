"""Built-in tool settings for advanced agents.

``resolve_built_in_tools`` turns an agent's tool settings into
``ResolvedBuiltInTools``, and the advanced agent builders apply it. Without
settings, the builders keep deepagents' own file-tool middleware, which also
applies a subagent spec's ``permissions``. Memory and the uipath-agents meta
prompt then read ``DEFAULT_BUILT_IN_TOOLS``, and ``eval`` depends on its
feature flag alone.

The rules:

- A tool missing from the settings is off, so a tool added later never turns
  on by itself for an agent someone configured.
- ``read_file`` is always on, because ``FilesystemMiddleware`` requires it.
- ``execute`` is never offered, because the advanced agent's backend has no shell.
- ``eval`` is only recorded here: the caller builds the code interpreter.
- ``task`` isn't a setting: the agent always gets it, and a ``task`` entry is
  ignored like any unknown tool.
- No tool takes settings yet, so a ``config`` is logged and ignored.
- Nothing in the settings fails a run. An unknown tool or setting is logged and
  ignored, because the designer can ship a value before the runtime knows it.

To add a built-in tool:

1. A deepagents file tool needs nothing here: it arrives through ``FsToolName``
   on a deepagents bump and lands in ``FILE_TOOL_NAMES`` and
   ``enabled_file_tools`` (see "When deepagents changes its tools" below).
   For any other tool, add its name to ``BUILT_IN_TOOL_NAMES`` and a field
   without a default to ``ResolvedBuiltInTools``, and set the field in
   ``resolve_built_in_tools`` with ``_is_enabled``. Report the field in
   uipath-agents' ``AdvancedAgentTools.applied_capabilities``, which feeds the
   AgentRun ``Capabilities`` telemetry.
2. Decide what agents without settings get: the designer shows them as having
   every tool. A tool that is always available is on in
   ``DEFAULT_BUILT_IN_TOOLS``, and is read from
   ``built_in_tools or DEFAULT_BUILT_IN_TOOLS`` where it is built. A tool behind
   a feature flag, like ``eval``, takes that default from its flag instead: the
   caller gives it to agents without settings while the flag is on, and to
   agents with settings only when the field is on too. ``eval_requested`` in
   ``DEFAULT_BUILT_IN_TOOLS`` is never read. A tool that comes from a middleware
   goes in ``shared_middleware``, so subagents get it as well.
3. Use the field in the uipath-agents meta prompt if the prompt describes the
   tool.
4. Add the tool to the Studio Web designer in the Agents repo: frontend-sw's
   ``config/capabilities-catalog.ts`` and ``hooks/useCapabilityText.ts``, then
   ``pnpm translations:extract``. The designer writes every tool it lists the
   first time someone changes a capability, so a tool it doesn't list is off for
   every agent changed after that. Update ``DESIGNER_TOOL_MAP`` in uipath-agents'
   tool-surface test and its copy in frontend-sw's ``capability-settings.test.ts``.

When deepagents changes its tools:

uipath-agents pins deepagents exactly, so a change arrives only through a bump
PR there, and its tool-surface tests fail when the tool set changes. That PR
handles each change below. Agents without settings, and every agent while
EnableAdvancedAgentBuiltInToolSettings is off, use deepagents' own file-tool
middleware, so they get deepagents' file tools as they are on their next run
after the bump ships.

- A new file tool lands in ``FILE_TOOL_NAMES`` on its own. Agents with settings
  have it off until someone turns it on. Agents without settings get it on the
  bump, before the designer shows it. To hold it back, give those agents
  ``build_file_tool_middleware`` too, with a default that leaves the tool out
  until its feature flag is on, the way ``eval`` waits for
  EnableAdvancedAgentCodeInterpreter. ``DEFAULT_BUILT_IN_TOOLS`` alone won't do:
  it lists every name in ``FILE_TOOL_NAMES``. This replaces deepagents' own
  middleware for those agents, so a subagent spec's ``permissions`` no longer
  apply to them. Then add the tool to the designer, as in "To add a built-in
  tool".
- A removed tool becomes unknown: its entries are logged and ignored, and every
  agent loses it on the bump. Remove it from the designer's catalog no later
  than the bump ships: until then the designer shows a switch that does
  nothing.
- A renamed tool loses its saved settings: the old entry is unknown and the new
  name is missing, so the tool is off. Read the old name's entry when the new
  name is missing, both in ``resolve_built_in_tools`` and in the designer's
  ``isCapabilityOn``, so saved agents keep their choice without being saved
  again. Rename it in the designer's catalog, in ``useCapabilityText`` and its
  en.json ids, and in both copies of ``DESIGNER_TOOL_MAP``.
- A tool that keeps its name but does more, through new arguments or wider
  effects, keeps every saved "on". No test catches this: read deepagents'
  changelog for each built-in tool on every bump. If a saved "on" shouldn't
  cover what the tool does now, hold the bump until the change can be gated.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import get_args

from deepagents import FilesystemMiddleware, FsToolName
from deepagents.backends import BackendProtocol
from langchain_core.language_models import BaseChatModel
from uipath.agent.models.agent import AdvancedAgentBuiltInToolSettings

from .code_interpreter import EVAL_TOOL_NAME

logger = logging.getLogger(__name__)

FILE_TOOL_NAMES: tuple[FsToolName, ...] = tuple(
    name for name in get_args(FsToolName) if name != "execute"
)
BUILT_IN_TOOL_NAMES = frozenset({*FILE_TOOL_NAMES, EVAL_TOOL_NAME})


@dataclass(frozen=True, kw_only=True)
class ResolvedBuiltInTools:
    """Which of deepagents' built-in tools and ``eval`` an agent gets.

    Attributes:
        enabled_file_tools: The deepagents file tools to build, in deepagents'
            order. A file tool not listed is never built, so neither the model
            nor the code interpreter can reach it.
        eval_requested: Whether the settings ask for ``eval``. uipath-langchain
            doesn't build the code interpreter; the caller does, and decides
            whether it can be built at all.
    """

    enabled_file_tools: tuple[FsToolName, ...]
    eval_requested: bool

    @property
    def supports_memory(self) -> bool:
        """Whether workspace memory can stay on: the model writes it with these tools."""
        return {"write_file", "edit_file"} <= set(self.enabled_file_tools)


DEFAULT_BUILT_IN_TOOLS = ResolvedBuiltInTools(
    enabled_file_tools=FILE_TOOL_NAMES, eval_requested=False
)
"""What an agent gets without built-in tool settings: every file tool. ``eval``
then depends on its feature flag alone."""


def resolve_built_in_tools(
    tool_settings: Mapping[str, AdvancedAgentBuiltInToolSettings],
) -> ResolvedBuiltInTools:
    """Resolve ``settings.advancedAgent.builtInTools`` into the tools to build.

    Args:
        tool_settings: The tool entries from agent.json, keyed by tool name.

    Returns:
        The tools the advanced agent builders apply.
    """
    for name in sorted(tool_settings.keys() - BUILT_IN_TOOL_NAMES):
        logger.warning("Ignoring unknown built-in tool %r", name)
    for name in sorted(tool_settings.keys() & BUILT_IN_TOOL_NAMES):
        for key in tool_settings[name].config:
            logger.warning("Ignoring unknown setting %r of built-in tool %r", key, name)
    if not _is_enabled(tool_settings, "read_file"):
        logger.warning("Built-in tool 'read_file' can't be turned off; keeping it on")

    return ResolvedBuiltInTools(
        enabled_file_tools=tuple(
            name
            for name in FILE_TOOL_NAMES
            if name == "read_file" or _is_enabled(tool_settings, name)
        ),
        eval_requested=_is_enabled(tool_settings, EVAL_TOOL_NAME),
    )


def _is_enabled(
    tool_settings: Mapping[str, AdvancedAgentBuiltInToolSettings], name: str
) -> bool:
    entry = tool_settings.get(name)
    return entry is not None and entry.enabled


def build_file_tool_middleware(
    built_in_tools: ResolvedBuiltInTools,
    model: BaseChatModel,
    backend: BackendProtocol | None,
) -> FilesystemMiddleware:
    """Build a file-tool middleware that offers only the enabled file tools.

    deepagents swaps a caller's middleware in for its own by name, so this one
    replaces deepagents' default. The tool descriptions come from the model's
    harness profile, as deepagents' own middleware would take them, so overrides
    such as ``read_file``'s still apply.

    Args:
        built_in_tools: The resolved built-in tools.
        model: The agent's chat model, whose harness profile holds the descriptions.
        backend: The agent's backend.

    Returns:
        A ``FilesystemMiddleware`` that builds only ``enabled_file_tools``.
    """
    from deepagents.profiles.harness.harness_profiles import (
        _harness_profile_for_model,
    )

    return FilesystemMiddleware(
        backend=backend,
        tools=list(built_in_tools.enabled_file_tools),
        custom_tool_descriptions=_harness_profile_for_model(
            model, None
        ).tool_description_overrides,
    )
