"""``UiPathSkills``: a coded agent's bound skill bundles, resolved once per process."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from uipath.platform import UiPath
from uipath.platform.skills import BundleFailure, SkillBundleRef, SkillSet

from .load_skill_tool import _validate_tool_name, create_load_skill_tool
from .prompt import skills_prompt_section

if TYPE_CHECKING:
    from deepagents.backends import BackendProtocol
    from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)

_NO_SOURCE = "none"
"""``SkillSet.source`` when no source was consulted: no bundles, or the SDK failed."""


class UiPathSkills:
    """Bound skill bundles for a coded LangGraph agent, resolved once per process.

    Declare the bundles at module level and load them in the graph factory,
    which the runtime calls once per process::

        SKILLS = UiPathSkills([SkillBundleRef(name="finance")])

        async def make_graph():
            await SKILLS.load()
            load_skill = SKILLS.load_skill_tool()
            prompt = BASE_PROMPT + "\\n\\n" + SKILLS.prompt_section()
            ...

    Nothing here needs credentials until ``load`` runs, so importing the agent
    module is cheap. ``load`` never raises for a bundle or a platform error:
    those land in ``SkillSet.failures`` and the agent runs without the skill.

    The set lives in this process only: a job resumed in a new process
    resolves the bundles again, so a ``latest`` bundle may come back newer.
    """

    def __init__(
        self,
        bundles: Sequence[SkillBundleRef],
        *,
        uipath_sdk: UiPath | None = None,
        destination: Path | None = None,
        tool_name: str = "load_skill",
    ) -> None:
        """Remember what to load; nothing is fetched and no SDK is built yet.

        Args:
            bundles: The skill bundles the agent is bound to.
            uipath_sdk: The SDK to load through; by default a ``UiPath()`` is
                built on the first ``load``.
            destination: The folder the bundles are written to; by default a
                temp folder the set owns and ``cleanup`` removes.
            tool_name: The name the model knows the load-skill tool by.

        Raises:
            ValueError: ``tool_name`` is not 1-64 letters, digits, ``_`` or ``-``.
        """
        _validate_tool_name(tool_name)
        self._bundles = tuple(bundles)
        self._sdk = uipath_sdk
        self._destination = destination
        self._tool_name = tool_name
        self._lock = asyncio.Lock()
        self._skill_set: SkillSet | None = None
        self._tool: tuple[SkillSet, StructuredTool] | None = None
        """The tool and the set it was built for."""

    @property
    def bundles(self) -> tuple[SkillBundleRef, ...]:
        """The declared bundle references."""
        return self._bundles

    @property
    def skill_set(self) -> SkillSet | None:
        """The resolved set; ``None`` until ``load`` has run."""
        return self._skill_set

    async def load(
        self, *, destination: Path | None = None, force: bool = False
    ) -> SkillSet:
        """Resolve the bundles, once per process.

        Concurrent calls share one resolution; later calls return the cached
        set. A platform error, including one building ``UiPath()``, is logged
        and cached as a set whose failures mark every bundle
        ``source_unavailable``, so the agent does not retry on every turn.
        ``force=True`` resolves again and removes the previous set's temp
        folder, if it owns one, so a tool built for that set stops working.

        Args:
            destination: Overrides the constructor's ``destination`` for this
                resolution.
            force: Resolve again even when a set is cached.

        Returns:
            The resolved set; never raises except on cancellation.
        """
        async with self._lock:
            if self._skill_set is not None and not force:
                return self._skill_set
            target = destination if destination is not None else self._destination
            result = await self._fetch(target)
            previous, self._skill_set, self._tool = self._skill_set, result, None
        if previous is not None and previous is not result and previous.owns_root:
            previous.cleanup()
        return result

    def prompt_section(self) -> str:
        """The ``## Skills`` system-prompt section; ``""`` before ``load`` or with no skills."""
        skill_set = self._skill_set
        if skill_set is None:
            return ""
        return skills_prompt_section(skill_set, tool_name=self._tool_name)

    def load_skill_tool(self) -> StructuredTool | None:
        """The ``load_skill`` tool for the model, built once per set.

        Its ``name`` argument is an enum of the set's qualified
        ``<bundle>/<skill>`` names. It is bound to that set and stops working
        after ``load(force=True)`` or ``cleanup``.

        Returns:
            The tool, or ``None`` when no bundle has a skill.

        Raises:
            RuntimeError: ``load`` has not run.
        """
        skill_set = self._skill_set
        if skill_set is None:
            raise RuntimeError("call await skills.load() first")
        if not skill_set.names():
            return None
        if self._tool is None or self._tool[0] is not skill_set:
            tool = create_load_skill_tool(skill_set, tool_name=self._tool_name)
            self._tool = (skill_set, tool)
        return self._tool[1]

    def advanced_middleware(
        self, backend: BackendProtocol
    ) -> list[AgentMiddleware[Any, Any]]:
        """The skills middleware for an advanced (deepagents) agent.

        Load with ``destination=<workspace>/skills``, where ``<workspace>`` is
        the backend's root, so the files live in the backend's filesystem at
        ``/skills/<bundle>/<skill>/``::

            backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)
            await skills.load(destination=workspace / "skills")
            agent = create_advanced_agent(
                model, middleware=skills.advanced_middleware(backend), backend=backend
            )

        The model reads skills with ``read_file``; pass ``middleware`` here
        instead of ``create_advanced_agent``'s ``skills`` argument.

        Args:
            backend: The workspace backend whose ``/skills`` folder holds the set.

        Returns:
            ``[UiPathSkillsMiddleware, SkillLoadTraceMiddleware]``, or ``[]``
            when no bundle has a skill.

        Raises:
            RuntimeError: ``load`` has not run, or the set's folder is gone
                (``cleanup`` ran).
        """
        skill_set = self._skill_set
        if skill_set is None:
            raise RuntimeError(
                "call await skills.load(destination=<workspace>/skills) first"
            )
        if not skill_set.names():
            return []
        if skill_set.root is None:
            raise RuntimeError(
                "the skill set has no folder (cleaned up); call "
                "await skills.load(destination=<workspace>/skills, force=True)"
            )
        if skill_set.root.name != "skills":
            logger.warning(
                "Skill bundles were loaded into %s; an advanced agent reads them "
                "at /skills/<bundle>/<skill>/, so load with "
                "destination=<workspace>/skills",
                skill_set.root,
            )
        from .middleware import SkillLoadTraceMiddleware, create_skills_middleware

        return [
            create_skills_middleware(backend, skill_set),
            SkillLoadTraceMiddleware(skill_set),
        ]

    def cleanup(self) -> None:
        """Remove the set's temp folder, if it owns one, and forget the set.

        Call it once ``load`` has finished, for example when the graph's
        context manager exits; the next ``load`` resolves again.
        """
        skill_set, self._skill_set, self._tool = self._skill_set, None, None
        if skill_set is not None:
            skill_set.cleanup()

    async def _fetch(self, target: Path | None) -> SkillSet:
        if not self._bundles:
            return SkillSet(source=_NO_SOURCE, bundles=[], failures=[])
        try:
            if self._sdk is None:
                self._sdk = UiPath()
            return await self._sdk.skills.load_all_async(
                list(self._bundles), destination=target
            )
        except Exception as e:
            return self._failed_set(e)

    def _failed_set(self, error: Exception) -> SkillSet:
        """A set that marks every bundle ``source_unavailable`` because of ``error``."""
        logger.warning(
            "Skill bundles could not be loaded; continuing without skills",
            exc_info=error,
        )
        return SkillSet(
            source=_NO_SOURCE,
            bundles=[],
            failures=[
                BundleFailure(ref=ref, reason="source_unavailable", detail=repr(error))
                for ref in self._bundles
            ],
        )
