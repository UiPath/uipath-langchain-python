"""``UiPathSkills``: a coded agent's bound skill bundles, resolved once per process."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import StructuredTool
from uipath.platform import UiPath
from uipath.platform.skills import BundleFailure, SkillBundleRef, SkillSet

from .load_skill_tool import _SKILL_MD, _validate_tool_name, create_load_skill_tool
from .prompt import skills_prompt_section

logger = logging.getLogger(__name__)

_NO_SOURCE = "none"
"""``SkillSet.source`` when no source was consulted: no bundles, or the SDK failed."""


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
        self._bundles = list(bundles)
        self._sdk = uipath_sdk
        self._destination = destination
        self._tool_name = tool_name
        self._skill_set: SkillSet | None = None
        self._tool: StructuredTool | None = None
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._sync_lock = threading.Lock()

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

        Args:
            destination: Overrides the constructor's ``destination`` for this call.
            force: Resolve again even when a set is cached.

        Returns:
            The resolved set; never raises except on cancellation.
        """
        if self._skill_set is not None and not force:
            return self._skill_set
        async with self._loop_lock():
            if self._skill_set is not None and not force:
                return self._skill_set
            target = destination if destination is not None else self._destination
            if not self._bundles:
                result = self._empty_set()
            else:
                try:
                    result = await self._sdk_or_new().skills.load_all_async(
                        self._bundles, destination=target
                    )
                except Exception as e:
                    result = self._failed_set(e)
            return self._store(result)

    def load_sync(
        self, *, destination: Path | None = None, force: bool = False
    ) -> SkillSet:
        """``load`` for sync code; safe to call from inside a running event loop.

        Args:
            destination: Overrides the constructor's ``destination`` for this call.
            force: Resolve again even when a set is cached.

        Returns:
            The resolved set, cached like ``load``'s; never raises.
        """
        if self._skill_set is not None and not force:
            return self._skill_set
        with self._sync_lock:
            if self._skill_set is not None and not force:
                return self._skill_set
            target = destination if destination is not None else self._destination
            if not self._bundles:
                result = self._empty_set()
            else:
                try:
                    result = self._sdk_or_new().skills.load_all(
                        self._bundles, destination=target
                    )
                except Exception as e:
                    result = self._failed_set(e)
            return self._store(result)

    def prompt_section(self) -> str:
        """The ``## Skills`` system-prompt section; ``""`` before ``load`` or with no skills."""
        if self._skill_set is None:
            return ""
        return skills_prompt_section(self._skill_set, tool_name=self._tool_name)

    def load_skill_tool(self) -> StructuredTool | None:
        """The ``load_skill`` tool for the model.

        After ``load`` this is the ``create_load_skill_tool`` tool for the set,
        built once and reused, or ``None`` when the set has no skills. Before
        ``load`` it is a lazy tool with the same name and metadata whose
        ``name`` argument is a plain string: its first call runs ``load`` and
        then answers exactly as the real tool does.
        """
        if self._skill_set is None:
            return self._lazy_tool()
        if not self._skill_set.names():
            return None
        return self._real_tool()

    def cleanup(self) -> None:
        """Remove the set's temp folder, if it owns one, and forget the set."""
        if self._skill_set is not None:
            self._skill_set.cleanup()
        self._skill_set = None
        self._tool = None

    def _loop_lock(self) -> asyncio.Lock:
        """The lock for the running loop; a new loop (``asyncio.run``) gets a new lock."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _sdk_or_new(self) -> UiPath:
        """The SDK to load through, built on first use so import needs no credentials."""
        if self._sdk is None:
            self._sdk = UiPath()
        return self._sdk

    def _store(self, result: SkillSet) -> SkillSet:
        """Cache ``result``, dropping the tool built for the previous set."""
        self._skill_set = result
        self._tool = None
        return result

    def _empty_set(self) -> SkillSet:
        return SkillSet(source=_NO_SOURCE, bundles=[], failures=[])

    def _failed_set(self, error: Exception) -> SkillSet:
        """A set that marks every bundle ``source_unavailable`` because of ``error``."""
        logger.warning(
            "Skill bundles could not be loaded; continuing without skills: %r", error
        )
        return SkillSet(
            source=_NO_SOURCE,
            bundles=[],
            failures=[
                BundleFailure(ref=ref, reason="source_unavailable", detail=repr(error))
                for ref in self._bundles
            ],
        )

    def _real_tool(self) -> StructuredTool:
        """The ``create_load_skill_tool`` tool for the loaded set, built once per set."""
        assert self._skill_set is not None
        if self._tool is None:
            self._tool = create_load_skill_tool(
                self._skill_set, tool_name=self._tool_name
            )
        return self._tool

    def _ensure_loaded_sync(self) -> None:
        """Load from sync code: on a fresh loop when none runs here, else via the SDK's sync twin."""
        if self._skill_set is not None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.load())
        else:
            self.load_sync()

    def _lazy_tool(self) -> StructuredTool:
        """A tool that loads the set on first call, then delegates to the real tool."""
        # A tool for an empty set has the real tool's description, metadata and
        # plain-string ``name`` schema; only its implementations are swapped.
        template = create_load_skill_tool(self._empty_set(), tool_name=self._tool_name)

        def load(name: str, path: str = _SKILL_MD) -> str:
            self._ensure_loaded_sync()
            func = self._real_tool().func
            assert func is not None
            return func(name, path)

        async def aload(name: str, path: str = _SKILL_MD) -> str:
            await self.load()
            coroutine = self._real_tool().coroutine
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
