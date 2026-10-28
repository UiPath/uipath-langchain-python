"""Middleware that gives an advanced (deepagents) agent its skills and traces their use.

``UiPathSkillsMiddleware`` lists the bound bundles to the model;
``SkillLoadTraceMiddleware`` emits a span each time the model reads one of
their ``SKILL.md`` files (DEC-31). Both work on a
``uipath.platform.skills.SkillSet`` materialized into the backend's
filesystem at ``/skills/<bundle>/<skill>/``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from deepagents.backends import BackendProtocol
from deepagents.backends.utils import validate_path

# The two listing helpers are private to deepagents; the pin (>=0.7.11,<0.8.0)
# keeps them stable, and importing them at module top makes a rename fail loudly.
from deepagents.middleware.skills import (
    SkillMetadata,
    SkillsMiddleware,
    SkillsState,
    SkillsStateUpdate,
    _alist_skills_with_errors,
    _list_skills_with_errors,
)
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import Command
from uipath.platform.skills import SkillSet

# Private SDK module on purpose: the skill span names and attributes are a
# cross-runtime contract, so they come from the SDK's one implementation.
from uipath.platform.skills._telemetry import record_skill_loaded

from .prompt import UIPATH_SKILLS_PROMPT

logger = logging.getLogger(__name__)

# deepagents' built-in filesystem tool; a SKILL.md it returns is a skill load.
_READ_FILE_TOOL = "read_file"


def deepagents_sources(skill_set: SkillSet) -> list[tuple[str, str]]:
    """deepagents ``SkillsMiddleware`` sources, one per bundle.

    Args:
        skill_set: The bundles materialized for this run.

    Returns:
        ``(virtual_root, label)`` pairs in bundle order, e.g.
        ``("/skills/finance/", "finance v3")``.
    """
    return [(b.virtual_root, b.label) for b in skill_set.bundles]


class UiPathSkillsMiddleware(SkillsMiddleware[Any, Any]):
    """deepagents' ``SkillsMiddleware`` for bound skill bundles.

    One labelled source per bundle and UiPath's prompt. The listing is
    overridden because deepagents keys skills by ``name`` across sources (last
    source wins): with bundles as the unit, and ``latest`` bundles that gain a
    skill after publish, a collision must never hide a skill. Every skill is
    kept and displayed as ``<bundle>/<skill>``. Reads resolve by ``path``, so
    the qualified display name changes nothing else.

    Reaches the main agent only: ``create_advanced_agent`` hands its
    ``middleware`` to the main agent alone. Subagents share the backend and can
    read a SKILL.md path the main agent hands them in the task instructions.
    """

    def __init__(self, *, backend: BackendProtocol, skill_set: SkillSet) -> None:
        """Initialize the middleware.

        Args:
            backend: The workspace backend whose ``/skills`` folder holds the
                materialized bundles.
            skill_set: The bundles written for this run, in binding order.
        """
        super().__init__(
            backend=backend,
            sources=deepagents_sources(skill_set),
            system_prompt=UIPATH_SKILLS_PROMPT,
        )
        self._bundles = [(b.ref.name, b.virtual_root) for b in skill_set.bundles]

    def _format_skills_locations(self) -> str:
        # Nothing is deduplicated, so deepagents' "(higher priority)" suffix on
        # the last source would only bias the model toward one bundle.
        return "\n".join(
            f"**{label} Skills**: `{path}`"
            for path, label in zip(self.sources, self.source_labels, strict=True)
        )

    def before_agent(  # type: ignore[override]
        self, state: SkillsState, runtime: Runtime[Any], config: RunnableConfig
    ) -> SkillsStateUpdate | None:
        """List every bundle's skills under qualified names, once per session.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            The listing, or ``None`` when ``skills_metadata`` is already in state.
        """
        if "skills_metadata" in state:
            return None
        skills: list[SkillMetadata] = []
        errors: list[str] = []
        for bundle_name, source_path in self._bundles:
            source_skills, source_error = _list_skills_with_errors(
                self._backend, source_path
            )
            _merge(bundle_name, source_skills, source_error, skills, errors)
        return _state_update(skills, errors)

    async def abefore_agent(  # type: ignore[override]
        self, state: SkillsState, runtime: Runtime[Any], config: RunnableConfig
    ) -> SkillsStateUpdate | None:
        """Async ``before_agent``."""
        if "skills_metadata" in state:
            return None
        skills: list[SkillMetadata] = []
        errors: list[str] = []
        for bundle_name, source_path in self._bundles:
            source_skills, source_error = await _alist_skills_with_errors(
                self._backend, source_path
            )
            _merge(bundle_name, source_skills, source_error, skills, errors)
        return _state_update(skills, errors)


def _merge(
    bundle_name: str,
    source_skills: list[SkillMetadata],
    source_error: str | None,
    into_skills: list[SkillMetadata],
    into_errors: list[str],
) -> None:
    if source_error is not None:
        into_errors.append(source_error)
    for skill in source_skills:
        renamed = SkillMetadata(**skill)
        renamed["name"] = f"{bundle_name}/{skill['name']}"
        into_skills.append(renamed)


def _state_update(skills: list[SkillMetadata], errors: list[str]) -> SkillsStateUpdate:
    update = SkillsStateUpdate(skills_metadata=skills)
    if errors:
        # Logged even when the prompt fragment is suppressed, as deepagents does.
        logger.warning("Skills load errors: %s", errors)
        update["skills_load_errors"] = errors
    return update


def create_skills_middleware(
    backend: BackendProtocol, skill_set: SkillSet
) -> UiPathSkillsMiddleware:
    """The skills middleware for a backend whose ``/skills`` folder holds ``skill_set``.

    Args:
        backend: The workspace backend the bundles were materialized into.
        skill_set: The bundles written for this run.

    Returns:
        A middleware listing one source per bundle.
    """
    return UiPathSkillsMiddleware(backend=backend, skill_set=skill_set)


class SkillLoadTraceMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Emit a ``Skill load`` span whenever the model reads a bound skill's ``SKILL.md``.

    deepagents has no load hook: a skill is "loaded" when ``read_file`` returns
    its ``SKILL.md``. Only the first page counts: ``read_file`` pages long
    files and the model continues with ``offset``. This wraps every tool call,
    so it stays cheap; tracing never raises and never alters the result, while
    exceptions from the tool itself propagate as usual.

    Main agent only (``create_advanced_agent`` does not forward the caller's
    middleware to subagents); a subagent's read of a SKILL.md shows up only as
    its ``read_file`` tool-call span. A ``read_file`` the model runs inside the
    code-interpreter sandbox does not pass through ``wrap_tool_call`` and
    emits no span.
    """

    def __init__(self, skill_set: SkillSet) -> None:
        """Initialize the middleware.

        Args:
            skill_set: The bundles written for this run.
        """
        self._skill_set = skill_set
        self._warned = False

    def _record(
        self, request: ToolCallRequest, result: ToolMessage | Command[Any]
    ) -> None:
        try:
            call = request.tool_call
            if (
                call["name"] != _READ_FILE_TOOL
                or not isinstance(result, ToolMessage)
                or result.status == "error"
            ):
                return
            args = call.get("args")
            if not isinstance(args, dict):
                return
            offset = args.get("offset", 0)
            if isinstance(offset, int) and offset > 0:
                return
            path = args.get("file_path")
            if not path:
                return
            # Normalize as the read_file tool does (backslashes, "//", "./").
            hit = self._skill_set.skill_at(validate_path(str(path)))
            if hit is None:
                return
            bundle, skill = hit
            record_skill_loaded(
                bundle,
                skill,
                size=len(result.text.encode("utf-8")),
                invoked_by="model",
            )
        except Exception:  # tracing must never affect the tool result
            # Warn once so a broken exporter is visible without flooding the log.
            logger.log(
                logging.DEBUG if self._warned else logging.WARNING,
                "Skill load tracing failed",
                exc_info=True,
            )
            self._warned = True

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run the tool, then trace the call if it read a bound ``SKILL.md``.

        Args:
            request: The tool call.
            handler: The next handler in the chain.

        Returns:
            The handler's result, unchanged.
        """
        result = handler(request)
        self._record(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async ``wrap_tool_call``."""
        result = await handler(request)
        self._record(request, result)
        return result
