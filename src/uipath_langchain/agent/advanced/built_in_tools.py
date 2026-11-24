"""Built-in tool settings for advanced agents.

``resolve_built_in_tools`` turns an agent's tool settings into
``ResolvedBuiltInTools``, and the advanced agent builders apply it with
``BuiltInToolFilterMiddleware``. deepagents still builds its own file-tool
middleware either way, so a subagent spec's ``permissions`` and the harness
profile's tool descriptions keep applying; the filter only hides the turned-off
tools and refuses them. Without settings there is no filter and memory isn't
gated; ``eval`` then depends on its feature flag alone. The uipath-agents meta
prompt reads ``DEFAULT_BUILT_IN_TOOLS`` for that case, so it must keep matching
what an agent without settings gets on a backend without a shell. Passing it as
``built_in_tools`` is not the same as passing nothing: that applies the filter.

The rules:

- A tool missing from the settings is off, so a tool added later never turns
  on by itself for an agent someone configured.
- ``read_file`` is always on: deepagents saves oversized tool results to files
  and tells the model to read them back with it, and attachments are read with
  it too.
- ``execute`` is never a setting. With settings, the filter hides and refuses it
  like every deepagents file tool that isn't enabled, so a backend with a shell
  can't bring it back. deepagents' grep description and shell-path prompt still
  mention it on such a backend, because they are written before the filter runs.
- ``eval`` is only recorded here: the caller builds the code interpreter.
- ``task`` isn't a setting: the agent always gets it, and a ``task`` entry is
  ignored like any unknown tool.
- No tool takes settings yet, so a ``config`` is logged and ignored.
- Nothing in the settings fails a run. An unknown tool or setting is logged and
  ignored, because the designer can ship a value before the runtime knows it.

To add a built-in tool:

1. A deepagents file tool needs nothing here: it arrives through ``FsToolName``
   on a deepagents bump and lands in ``FILE_TOOL_NAMES`` and in
   ``DEFAULT_BUILT_IN_TOOLS`` (see "When deepagents changes its tools" below).
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
   ``DEFAULT_BUILT_IN_TOOLS`` is never read. The filter only hides file tools:
   any other tool is turned off by not building it. A tool that comes from a
   middleware goes in ``shared_middleware``, so subagents get it as well.
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
EnableAdvancedAgentBuiltInToolSettings is off, get no ``built_in_tools`` and so
no filter: they get deepagents' file tools as they are on their next run after
the bump ships.

- A new file tool lands in ``FILE_TOOL_NAMES`` on its own. If a backend only
  offers it when it has some capability, as with ``execute``, leave it out of
  ``FILE_TOOL_NAMES`` instead: the filter still hides it. Agents with settings
  have it off until someone turns it on. Agents without settings get it on the
  bump, before the designer shows it. To hold it back, pass those agents a
  ``built_in_tools`` that leaves the tool out until its feature flag is on, the
  way ``eval`` waits for EnableAdvancedAgentCodeInterpreter.
  ``DEFAULT_BUILT_IN_TOOLS`` alone won't do: it lists every name in
  ``FILE_TOOL_NAMES``. Passing ``built_in_tools`` also hides ``execute``, gates
  their memory, and rejects compiled and remote subagents and a caller's own
  subagent middleware. Then add the tool to the designer, as
  in "To add a built-in tool".
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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, get_args

from deepagents import FsToolName
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from uipath.agent.models.agent import AdvancedAgentBuiltInToolSettings

from .code_interpreter import EVAL_TOOL_NAME

logger = logging.getLogger(__name__)

FILE_TOOL_NAMES: tuple[FsToolName, ...] = tuple(
    name for name in get_args(FsToolName) if name != "execute"
)
"""The deepagents file tools that are settings: every ``FsToolName`` but
``execute``, which a backend offers only when it has a shell."""

_FILTERED_TOOL_NAMES = frozenset(get_args(FsToolName))
"""What the filter decides on: every deepagents file tool, ``execute`` included,
so one the settings don't enable is hidden even when it isn't a setting."""

BUILT_IN_TOOL_NAMES = frozenset({*FILE_TOOL_NAMES, EVAL_TOOL_NAME})


@dataclass(frozen=True, kw_only=True)
class ResolvedBuiltInTools:
    """Which of deepagents' built-in tools and ``eval`` an agent gets.

    Attributes:
        enabled_file_tools: The deepagents file tools the agent may use, in
            deepagents' order. deepagents still builds the others;
            ``BuiltInToolFilterMiddleware`` hides them from the model and the
            code interpreter, and refuses calls to them.
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
then depends on its feature flag alone. Keep it equal to what the builders give
for ``built_in_tools=None`` on a backend without a shell: the uipath-agents meta
prompt reads it for that case. Passing it as ``built_in_tools`` still applies the
filter, which also hides ``execute``."""


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


class BuiltInToolFilterMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Hide the file tools an agent's settings turn off, and refuse calls to them.

    deepagents' own file-tool middleware still builds every file tool, with the
    subagent's ``permissions`` and the harness profile's descriptions. Listed
    ahead of the code interpreter, this also keeps the turned-off tools out of
    ``eval``, which reads the same ``request.tools``.
    """

    def __init__(self, built_in_tools: ResolvedBuiltInTools) -> None:
        self.turned_off = _FILTERED_TOOL_NAMES - set(built_in_tools.enabled_file_tools)

    def _visible(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        tools = [
            tool for tool in request.tools if _tool_name(tool) not in self.turned_off
        ]
        return request.override(tools=tools)

    def _narrowed(self, request: ToolCallRequest) -> ToolCallRequest:
        """Leave the turned-off tools out of what inner middleware can switch to.

        An approval step that lets a reviewer edit a call looks the edited tool up
        in ``runtime.tools``, after this hook has checked the original name.
        """
        runtime = request.runtime
        tools = [tool for tool in runtime.tools if tool.name not in self.turned_off]
        return replace(request, runtime=replace(runtime, tools=tools))

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        name = request.tool_call["name"]
        if name not in self.turned_off:
            return None
        return ToolMessage(
            content=f"Error: the {name} tool is turned off for this agent.",
            tool_call_id=request.tool_call["id"],
            name=name,
            status="error",
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._visible(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._visible(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._refusal(request) or handler(self._narrowed(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return self._refusal(request) or await handler(self._narrowed(request))


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    return tool.name if isinstance(tool, BaseTool) else tool.get("name")
