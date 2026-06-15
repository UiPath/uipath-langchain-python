"""LangGraph executor for MCP-backed UiPath jobs.

When an MCP ``tools/call`` starts a UiPath job (the server returns a
``uipath.com/job`` handle), :class:`LangGraphJobExecutor` suspends the LangGraph
agent on that job and resumes when it finishes — the same durable suspend/resume
mechanism :mod:`uipath_langchain.agent.tools.process_tool` uses.

It mirrors ``process_tool`` exactly:

* the START ``tools/call`` runs **inside** ``@durable_interrupt`` so it executes
  exactly once (a resume re-runs the node but skips the body);
* it interrupts with ``WaitJobRaw`` so the runtime persists a ``JOB`` resume
  trigger (with the ``JOB_RAW`` name → resume without output extraction) and
  Orchestrator resumes the agent when the child job reaches a terminal state;
* on resume the body is skipped and ``interrupt(None)`` returns the terminal
  ``Job``; we re-derive the ``{key, folderKey}`` handle from it and FETCH the
  result with a follow-up ``tools/call`` (the server formats the output).

The neutral wire work (building the START/FETCH ``_meta``, parsing the handle)
lives in the sibling :mod:`~uipath_langchain.agent.tools.mcp.jobs` module; this class only owns the *suspend*
policy, so ``interrupt`` is confined here.
"""

from __future__ import annotations

from typing import Any

from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator import Job, JobState

from uipath_langchain._utils.durable_interrupt import (
    SkipInterruptValue,
    durable_interrupt,
)

from .jobs import (
    FetchFn,
    JobStart,
    StartFn,
    UiPathJobHandle,
)


class _NonJobStartValue(SkipInterruptValue):
    """Carries a non-job :class:`JobStart` back through ``@durable_interrupt``.

    A job-aware client sends the START ``_meta`` on every call, but the server
    only returns a handle for job-backed tools. For a normal (non-job) result we
    must NOT suspend — yet the ``@durable_interrupt`` body still has to run on
    every pass to keep the durable index aligned. Returning this value injects the
    result into the scratchpad and resumes immediately, without a real suspend.
    """

    def __init__(self, outcome: JobStart) -> None:
        self._outcome = outcome

    @property
    def resume_value(self) -> Any:
        """The :class:`JobStart` to return to the executor without suspending."""
        return self._outcome


class LangGraphJobExecutor:
    """``McpJobExecutor`` that suspends the LangGraph agent on the started job."""

    async def run(self, *, start: StartFn, fetch: FetchFn, tool_name: str) -> Any:
        """Start the job, suspend until it finishes, then FETCH its result.

        Args:
            start: Issues the START ``tools/call`` once; returns a :class:`JobStart`.
            fetch: Re-calls the tool with the FETCH ``_meta`` for a handle.
            tool_name: The MCP tool name (for diagnostics).

        Returns:
            The FETCH result for a job-backed call, or the normal tool result when
            the call did not start a job.
        """

        @durable_interrupt
        async def _suspend_on_job() -> Any:
            outcome = await start()
            if outcome.handle is None:
                # Non-job tool / no opt-in: do not suspend; carry the result.
                return _NonJobStartValue(outcome)
            # The whole handle round-trips the suspend via this WaitJobRaw payload.
            return WaitJobRaw(
                job=Job(
                    id=0,
                    key=outcome.handle.job_key,
                    folder_key=outcome.handle.folder_key,
                ),
                process_folder_key=outcome.handle.folder_key,
            )

        resumed = await _suspend_on_job()

        if isinstance(resumed, JobStart):
            # Non-job path: the START result is the tool's output.
            return resumed.result

        # Resume path: `resumed` is the runtime-materialized terminal raw Job.
        # WaitJobRaw skips state validation, so re-check for failure (as process_tool does).
        job = resumed
        if (job.state or "").lower() == JobState.FAULTED:
            return str(
                getattr(job, "info", None) or f"Job for tool '{tool_name}' faulted"
            )
        if not job.key or not job.folder_key:
            return str(
                getattr(job, "info", None)
                or f"Job for tool '{tool_name}' returned no key to fetch its result"
            )

        # Re-derive the handle from the resumed Job and let the server format the result.
        handle = UiPathJobHandle(job_key=job.key, folder_key=job.folder_key)
        return await fetch(handle)
