"""Framework-neutral helpers for long-running UiPath jobs over MCP.

Self-contained in uipath-langchain (no base-SDK change): the ``uipath.com/job``
``_meta`` contract helpers + the executor abstraction. The base SDK already provides
the generic durable primitives this builds on (``WaitJobRaw`` and the jobs service);
only the MCP-specific glue lives here.

``interrupt`` / langgraph stay out of this module — they live in
:mod:`uipath_langchain.agent.tools.mcp.job_executor` (``LangGraphJobExecutor``). The
``_meta`` helpers operate on plain dicts so they are MCP-SDK-version agnostic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from uipath.platform.orchestrator import Job, JobState

__all__ = [
    "JOB_META_KEY",
    "JOB_PROTOCOL_VERSION",
    "BlockingJobExecutor",
    "FetchFn",
    "JobStart",
    "JobStatusReader",
    "McpJobExecutor",
    "StartFn",
    "UiPathJobHandle",
    "build_fetch_meta",
    "build_start_meta",
    "read_job_handle",
    "read_job_version",
]

JOB_META_KEY = "uipath.com/job"
"""Reverse-DNS ``_meta`` key under which all job signaling lives."""

JOB_PROTOCOL_VERSION = 1
"""Current ``uipath.com/job`` contract version emitted by this client."""


@dataclass(frozen=True)
class UiPathJobHandle:
    """Handle to a UiPath job started behind an MCP ``tools/call``.

    Returned by the server in the START response ``_meta`` and used to suspend on
    the job and to FETCH its result.
    """

    job_key: str
    folder_key: str


@dataclass(frozen=True)
class JobStart:
    """Outcome of the START ``tools/call``: a job handle, or a normal tool result."""

    handle: Optional[UiPathJobHandle]
    result: Any = None


def build_start_meta(version: int = JOB_PROTOCOL_VERSION) -> Dict[str, Any]:
    """Build the START opt-in ``_meta`` (no ``key`` ⇒ START intent)."""
    return {JOB_META_KEY: {"version": version}}


def build_fetch_meta(handle: UiPathJobHandle) -> Dict[str, Any]:
    """Build the FETCH ``_meta`` for a started job (``key`` present ⇒ FETCH intent)."""
    return {JOB_META_KEY: {"key": handle.job_key, "folderKey": handle.folder_key}}


def _job_section(meta: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not meta:
        return None
    section = meta.get(JOB_META_KEY)
    return section if isinstance(section, Mapping) else None


def read_job_handle(meta: Optional[Mapping[str, Any]]) -> Optional[UiPathJobHandle]:
    """Parse a job handle from a result's ``_meta`` mapping.

    Returns a :class:`UiPathJobHandle` when both ``key`` and ``folderKey`` are
    present (a START response), else ``None`` (a normal result / version-only opt-in).
    """
    section = _job_section(meta)
    if not section:
        return None
    key = section.get("key")
    folder_key = section.get("folderKey")
    if isinstance(key, str) and key and isinstance(folder_key, str) and folder_key:
        return UiPathJobHandle(job_key=key, folder_key=folder_key)
    return None


def read_job_version(meta: Optional[Mapping[str, Any]]) -> Optional[int]:
    """Parse the advertised / opted-in contract version from a ``_meta`` mapping."""
    section = _job_section(meta)
    if not section:
        return None
    version = section.get("version")
    return version if isinstance(version, int) else None


StartFn = Callable[[], Awaitable[JobStart]]
"""Issues the START ``tools/call`` once and returns its :class:`JobStart` outcome."""

FetchFn = Callable[[UiPathJobHandle], Awaitable[Any]]
"""Re-calls the tool with the FETCH ``_meta`` for a handle; returns the job result."""

_TERMINAL_STATES = frozenset({JobState.SUCCESSFUL.value, JobState.FAULTED.value})


@runtime_checkable
class McpJobExecutor(Protocol):
    """Awaits a job-backed MCP tool call and returns its final output.

    An implementation owns the START → await → FETCH lifecycle for one tool call:
    it invokes ``start`` (exactly once, inside its durable boundary when it
    suspends), waits for the job to finish, then returns ``await fetch(handle)``.
    Implementations differ only in *how* they wait (suspend vs poll).
    """

    async def run(self, *, start: StartFn, fetch: FetchFn, tool_name: str) -> Any:
        """Run one job-backed tool call to completion."""
        ...


@runtime_checkable
class JobStatusReader(Protocol):
    """Minimal jobs-service shape consumed by :class:`BlockingJobExecutor`."""

    async def retrieve_async(
        self, job_key: str, *, folder_key: Optional[str] = None
    ) -> Job:
        """Retrieve the job identified by ``job_key`` in folder ``folder_key``."""
        ...


class BlockingJobExecutor:
    """Neutral default executor: poll the job to a terminal state, then FETCH.

    Does **not** suspend the host — correct in any environment (a CLI, an eval
    harness, a framework without durable interrupts). The child job stays running
    while we poll, but the tool always returns the right result. Hosts that *can*
    suspend should inject a framework-specific executor instead (e.g.
    :class:`~uipath_langchain.agent.tools.mcp.job_executor.LangGraphJobExecutor`).
    """

    def __init__(
        self,
        jobs: Optional[JobStatusReader] = None,
        *,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize the executor.

        Args:
            jobs: A jobs service exposing ``retrieve_async(job_key, *, folder_key)``.
                Defaults to ``UiPath().jobs`` (constructed lazily) when ``None``.
            poll_interval: Seconds between status polls.
            timeout: Optional overall timeout in seconds; ``None`` waits forever.
        """
        self._jobs = jobs
        self._poll_interval = poll_interval
        self._timeout = timeout

    def _jobs_service(self) -> JobStatusReader:
        if self._jobs is None:
            from uipath.platform import UiPath

            self._jobs = UiPath().jobs
        return self._jobs

    async def run(self, *, start: StartFn, fetch: FetchFn, tool_name: str) -> Any:
        """Start the job, poll until terminal, then FETCH its result."""
        outcome = await start()
        if outcome.handle is None:
            return outcome.result
        await self._wait_until_terminal(outcome.handle)
        return await fetch(outcome.handle)

    async def _wait_until_terminal(self, handle: UiPathJobHandle) -> None:
        jobs = self._jobs_service()
        loop = asyncio.get_event_loop()
        deadline = None if self._timeout is None else loop.time() + self._timeout
        while True:
            job = await jobs.retrieve_async(
                handle.job_key, folder_key=handle.folder_key
            )
            if (job.state or "").lower() in _TERMINAL_STATES:
                return
            if deadline is not None and loop.time() >= deadline:
                raise TimeoutError(
                    f"Job {handle.job_key} did not reach a terminal state "
                    f"within {self._timeout}s"
                )
            await asyncio.sleep(self._poll_interval)
