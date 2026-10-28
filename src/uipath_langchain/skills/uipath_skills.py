"""``UiPathSkills``: a coded agent's bound skill bundles, resolved once per process."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from langchain_core.tools import StructuredTool
from uipath.platform import UiPath
from uipath.platform.skills import BundleFailure, SkillBundleRef, SkillSet

from .load_skill_tool import _SKILL_MD, _validate_tool_name, create_load_skill_tool
from .prompt import skills_prompt_section

if TYPE_CHECKING:
    from deepagents.backends import BackendProtocol
    from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)

_NO_SOURCE = "none"
"""``SkillSet.source`` when no source was consulted: no bundles, or the SDK failed."""


class _Claim(NamedTuple):
    """What a ``load`` caller does next: return ``cached``, or resolve / wait on ``future``."""

    cached: SkillSet | None
    future: Future[SkillSet | None] | None
    owner: bool
    owner_loop: asyncio.AbstractEventLoop | None
    generation: int
    """``UiPathSkills._generation`` at claim time; ``cleanup`` moves it on."""


class UiPathSkills:
    """Bound skill bundles for a coded LangGraph agent, resolved once per process.

    Declare the bundles at module level, then hand the model the tool and the
    prompt text::

        skills = UiPathSkills([SkillBundleRef(name="finance")])

        async def agent(state):
            await skills.load()
            prompt = BASE_PROMPT + "\\n\\n" + skills.prompt_section()
            tools = [t for t in [skills.load_skill_tool()] if t is not None]
            ...

    Nothing here needs credentials until ``load`` runs, so importing the agent
    module is cheap. ``load`` never raises for a bundle or a platform error:
    those land in ``SkillSet.failures`` and the agent runs without the skill.
    Concurrent first calls, from any mix of threads, event loops, ``load`` and
    ``load_sync``, share one resolution.

    The set lives in this process only: a job resumed in a new process
    resolves the bundles again, so a ``latest`` bundle may come back newer.

    A real tool from ``load_skill_tool`` (after ``load``) is bound to the set
    it was built for and stops working after ``load(force=True)`` or
    ``cleanup``; the lazy tool (before ``load``) always follows the current set.
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
        self._lock = threading.Lock()
        """Guards the SDK and every field below."""
        self._skill_set: SkillSet | None = None
        self._tool: tuple[SkillSet, StructuredTool] | None = None
        """The real tool and the set it was built for."""
        self._inflight: Future[SkillSet | None] | None = None
        """The resolution in progress; its result is ``None`` when it was abandoned."""
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        """The loop the in-flight resolution runs on; ``None`` for a sync one."""
        self._generation = 0
        """Bumped by ``cleanup``: a resolution claimed before it is not stored."""

    @property
    def bundles(self) -> tuple[SkillBundleRef, ...]:
        """The declared bundle references."""
        return self._bundles

    @property
    def skill_set(self) -> SkillSet | None:
        """The resolved set; ``None`` until ``load`` or ``load_sync`` has run."""
        return self._skill_set

    async def load(
        self, *, destination: Path | None = None, force: bool = False
    ) -> SkillSet:
        """Resolve the bundles, once per process.

        Concurrent first calls share one resolution; later calls return the
        cached set. A platform error, including one building ``UiPath()``, is
        logged and cached as a set whose failures mark every bundle
        ``source_unavailable``, so the agent does not retry on every turn.
        ``force=True`` resolves again (or joins a resolution in progress); the
        previous set's temp folder, if it owns one, is then removed, so a real
        tool from ``load_skill_tool`` built for it stops working.

        Args:
            destination: Overrides the constructor's ``destination``; applies
                only when this call actually runs a resolution.
            force: Resolve again even when a set is cached.

        Returns:
            The resolved set; never raises except on cancellation.
        """
        target = destination if destination is not None else self._destination
        while True:
            claim = self._claim(force, asyncio.get_running_loop())
            if claim.cached is not None:
                return claim.cached
            assert claim.future is not None
            if claim.owner:
                try:
                    result = await self._fetch_async(target)
                except BaseException:
                    self._abandon(claim.future)
                    raise
                return self._finish(claim.future, result, claim.generation)
            waited = await asyncio.wrap_future(claim.future)
            if waited is not None:
                return waited
            # The owner was cancelled before it finished: claim again.

    def load_sync(
        self, *, destination: Path | None = None, force: bool = False
    ) -> SkillSet:
        """``load`` for sync code; safe to call from inside a running event loop.

        Called on the event loop that is running ``load`` (blocking on it
        would deadlock), it resolves separately into a fresh temp folder and
        returns that set uncached: the caller owns it and should call its
        ``cleanup`` when done.

        Args:
            destination: Overrides the constructor's ``destination``; applies
                only when this call actually runs a shared resolution.
            force: Resolve again even when a set is cached.

        Returns:
            The resolved set, cached and shared like ``load``'s except in the
            case above; never raises except on ``KeyboardInterrupt`` and the
            like.
        """
        return self._load_sync(destination, force)[0]

    def _load_sync(
        self, destination: Path | None, force: bool
    ) -> tuple[SkillSet, bool]:
        """``load_sync``, plus whether the set is the shared one (``False``: caller cleans up)."""
        target = destination if destination is not None else self._destination
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        while True:
            claim = self._claim(force, None)
            if claim.cached is not None:
                return claim.cached, True
            assert claim.future is not None
            if claim.owner:
                try:
                    result = self._fetch_sync(target)
                except BaseException:
                    self._abandon(claim.future)
                    raise
                return self._finish(claim.future, result, claim.generation), True
            if loop is not None and claim.owner_loop is loop:
                # The resolution in progress runs on this thread's loop, which
                # blocking here would starve: resolve separately, uncached, into
                # a fresh temp folder so the owner's folder is never emptied.
                logger.warning(
                    "load_sync called on the event loop that is running load(); "
                    "resolving the skill bundles separately"
                )
                return self._fetch_sync(None), False
            waited = claim.future.result()
            if waited is not None:
                return waited, True

    def prompt_section(self) -> str:
        """The ``## Skills`` system-prompt section; ``""`` before ``load`` or with no skills."""
        skill_set = self._skill_set
        if skill_set is None:
            return ""
        return skills_prompt_section(skill_set, tool_name=self._tool_name)

    def load_skill_tool(self) -> StructuredTool | None:
        """The ``load_skill`` tool for the model.

        After ``load`` this is the ``create_load_skill_tool`` tool for the set,
        built once and reused, or ``None`` when the set has no skills; it is
        bound to that set and stops working after ``load(force=True)`` or
        ``cleanup``. Before ``load`` it is a lazy tool with the same name and
        metadata whose ``name`` argument is a plain string: each call loads (or
        reuses) the current set and then answers exactly as the real tool does.
        """
        skill_set = self._skill_set
        if skill_set is None:
            return self._lazy_tool()
        if not skill_set.names():
            return None
        return self._tool_for(skill_set)

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

        A resolution in progress is not stored when it finishes: its set is
        cleaned up and handed only to the ``load`` call that ran it; calls
        waiting on it resolve again.
        """
        with self._lock:
            skill_set = self._skill_set
            self._skill_set = None
            self._tool = None
            self._generation += 1
        if skill_set is not None:
            skill_set.cleanup()

    def _claim(self, force: bool, loop: asyncio.AbstractEventLoop | None) -> _Claim:
        """Return the cached set, join the resolution in progress, or start one."""
        with self._lock:
            generation = self._generation
            if self._inflight is not None:
                return _Claim(None, self._inflight, False, self._owner_loop, generation)
            if self._skill_set is not None and not force:
                return _Claim(self._skill_set, None, False, None, generation)
            future: Future[SkillSet | None] = Future()
            # A running future cannot be cancelled, so a waiter whose
            # ``wrap_future`` is cancelled does not cancel it for the others.
            future.set_running_or_notify_cancel()
            self._inflight = future
            self._owner_loop = loop
            return _Claim(None, future, True, loop, generation)

    def _finish(
        self, future: Future[SkillSet | None], result: SkillSet, generation: int
    ) -> SkillSet:
        """Store ``result``, release the waiters and remove the replaced set's temp folder.

        When ``cleanup`` ran since the claim (``generation`` moved), ``result``
        is cleaned up instead of stored and the waiters claim again.
        """
        with self._lock:
            stale = generation != self._generation
            if stale:
                self._inflight = None
                self._owner_loop = None
        if stale:
            future.set_result(None)
            result.cleanup()
            return result
        with self._lock:
            previous = self._skill_set
            self._skill_set = result
            self._tool = None
            self._inflight = None
            self._owner_loop = None
        future.set_result(result)
        if previous is not None and previous is not result and previous.owns_root:
            previous.cleanup()
        return result

    def _abandon(self, future: Future[SkillSet | None]) -> None:
        """Let the waiters of a cancelled resolution claim again."""
        with self._lock:
            self._inflight = None
            self._owner_loop = None
        future.set_result(None)

    async def _fetch_async(self, target: Path | None) -> SkillSet:
        if not self._bundles:
            return self._empty_set()
        try:
            return await self._sdk_or_new().skills.load_all_async(
                list(self._bundles), destination=target
            )
        except Exception as e:
            return self._failed_set(e)

    def _fetch_sync(self, target: Path | None) -> SkillSet:
        if not self._bundles:
            return self._empty_set()
        try:
            return self._sdk_or_new().skills.load_all(
                list(self._bundles), destination=target
            )
        except Exception as e:
            return self._failed_set(e)

    def _sdk_or_new(self) -> UiPath:
        """The SDK to load through, built on first use so import needs no credentials."""
        with self._lock:
            if self._sdk is None:
                self._sdk = UiPath()
            return self._sdk

    def _empty_set(self) -> SkillSet:
        return SkillSet(source=_NO_SOURCE, bundles=[], failures=[])

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

    def _tool_for(self, skill_set: SkillSet) -> StructuredTool:
        """The real tool for ``skill_set``, built once while it is the current set."""
        with self._lock:
            if self._tool is not None and self._tool[0] is skill_set:
                return self._tool[1]
        tool = create_load_skill_tool(skill_set, tool_name=self._tool_name)
        with self._lock:
            if skill_set is not self._skill_set:
                return tool
            if self._tool is not None and self._tool[0] is skill_set:
                return self._tool[1]
            self._tool = (skill_set, tool)
        return tool

    def _lazy_tool(self) -> StructuredTool:
        """A tool that loads the current set on each call, then delegates to its real tool."""
        # A tool for an empty set has the real tool's description, metadata and
        # plain-string ``name`` schema; only its implementations are swapped.
        template = create_load_skill_tool(self._empty_set(), tool_name=self._tool_name)

        def load(name: str, path: str = _SKILL_MD) -> str:
            skill_set, shared = self._load_sync(None, False)
            try:
                func = self._tool_for(skill_set).func
                assert func is not None
                return func(name, path)
            finally:
                if not shared:
                    # The separate same-loop resolution: nobody else holds it.
                    skill_set.cleanup()

        async def aload(name: str, path: str = _SKILL_MD) -> str:
            coroutine = self._tool_for(await self.load()).coroutine
            assert coroutine is not None
            return await coroutine(name, path)

        return StructuredTool.from_function(
            func=load,
            coroutine=aload,
            name=template.name,
            args_schema=template.args_schema,
            description=template.description,
            metadata=dict(template.metadata or {}),
        )
