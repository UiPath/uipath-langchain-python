"""Suspend the run while a long MCP task finishes (``com.uipath/tasks``).

``com.uipath/tasks`` is the UiPath name of the MCP Tasks protocol: same
``resultType: "task"``, same ``tasks/get``. A UiPath MCP server that serves it
also says, on the task, which Orchestrator job is behind it. A caller that is
itself an Orchestrator job can then stop holding compute while that job runs:
it suspends on a Job-type resume trigger and is started again when the job
ends. Polling (``mcp_tasks.py``) keeps the run awake for the whole wait;
this module is what lets it sleep.

Suspending is not free -- a resume starts the runtime again, and the trigger
can take minutes to fire -- so it is not done at once. The task is polled for
``minPollingTimeBeforeSuspension`` first, and the run suspends only if it is
still working after that. A call that finishes inside that time never costs a
suspend. How long that is comes, in this order, from the caller's own setting
(the ``McpTasksMinPollingTimeBeforeSuspensionSeconds`` flag), from the server
(``suspendAfterMs`` on the job annotation), or from the default of a minute
(:func:`polling_time_before_suspension`).

How a suspended call survives the resume. On resume the tool node runs again
from its first line, so every step that must not happen twice is a
``@durable_interrupt`` step, whose saved value is returned instead of running
its body:

1. ``submit`` sends ``tools/call`` once. For a task it saves the task handle
   -- the task id, the poll interval, the job annotation and when the call
   was sent. That is everything needed to go on polling ``tasks/get`` after a
   resume, in a new process, on a new connection.
2. ``park``, only for a task that names a job, polls the task for whatever is
   left of the polling time and, if it is still working, returns the
   ``WaitJobRaw`` that suspends the run on the job. The polling is inside the
   step so that a resumed run does not do it again.
3. ``collect`` polls the task to its end: after the resume, or from the start
   for a task that names no job.

Whichever step ends the call saves the result with it (``save_results``), so
that a coded node which suspends later on something else gets it back on the
replay: the call is not sent again and the task is not polled again, which
matters once the server no longer keeps it. A tool node runs one call and
nothing after it, so it saves only the handle.

After the resume the task is read with ``tasks/get`` like any other poll, so
the result still comes through the server and its guardrail screening. The
resume trigger only says *when* to look; it is never the source of the result.

The steps run in that order on every pass, which keeps their indexes in
step with LangGraph's interrupt counter. Which of them run is decided by what
``submit`` and ``park`` saved, so a replay takes the same path as the first
pass. The kill switch does not change the path of a call that is
already under way: a call being replayed stays on these steps
(:func:`durable_call_in_play`), and the switch only stops ``park`` from
starting a new suspend.

One limit, from ``@durable_interrupt`` itself: its steps are counted from the
start of the node, and LangGraph's own ``interrupt()`` is not part of that
count. A node that calls ``interrupt()`` *before* a call made here cannot
resume it. In a coded agent, make the call first or give the interrupt its own
node.
"""

import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, TypeVar

import anyio
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph.config import get_config
from uipath.core.feature_flags import FeatureFlags
from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator.job import Job
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils.durable_interrupt import (
    SkipInterruptValue,
    durable_interrupt,
)
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

logger: logging.Logger = logging.getLogger(__name__)

UIPATH_TASKS_EXTENSION_ID = "com.uipath/tasks"

#: The ``_meta`` key on a task that names the Orchestrator job behind it.
UIPATH_TASK_META_KEY = "com.uipath/task"

#: The version of that annotation read here.
UIPATH_TASK_ANNOTATION_VERSION = 1

#: The resume-trigger kind a run can suspend on here. The server may name other
#: kinds later; a task annotated with one of those is polled.
JOB_TRIGGER_TYPE = "Job"

#: The ``com.uipath/tasks`` declaration: the trigger kinds this client handles.
#: A server names a job only to a client whose list includes ``Job``.
UIPATH_TASKS_CAPABILITY: dict[str, Any] = {"resumeTriggerTypes": [JOB_TRIGGER_TYPE]}

#: Kill switch; set it to ``true`` to keep polling and never suspend.
SUSPENSION_KILL_SWITCH = "DisableMcpTasksSuspension"

#: ``minPollingTimeBeforeSuspension``, in seconds.
MIN_POLLING_TIME_FLAG = "McpTasksMinPollingTimeBeforeSuspensionSeconds"

# A suspend costs a runtime start on the way back and waits on a trigger that
# is not instant, so it only pays for a job that runs for a while. There is
# nothing to tell a long job from a short one but time: a task still working
# after this long is taken to be a long one. The same minute the long-running
# A2A call polls for before it suspends.
DEFAULT_MIN_POLLING_TIME_BEFORE_SUSPENSION = 60.0

_PAYLOAD_KEY = "__mcp_task_suspension__"

ResultT = TypeVar("ResultT")
HandleT = TypeVar("HandleT")


class PollWindowElapsed(Exception):
    """A bounded wait on a task ended with the task still working."""


@dataclass(frozen=True)
class SuspensionSettings:
    """When a task call suspends the run.

    Attributes:
        min_polling_time_before_suspension: Seconds. How long a task is polled
            before the run suspends on its job. Zero suspends without polling.
            ``None`` leaves it to the server, and to the default when the
            server names none (:func:`polling_time_before_suspension`).
    """

    min_polling_time_before_suspension: float | None = None


@dataclass(frozen=True)
class TaskJob:
    """The Orchestrator job behind a task, from its ``com.uipath/task`` annotation.

    Attributes:
        job_key: The job to suspend on.
        folder_key: The folder the job runs in. The resume reads the job
            there, so a job named without one is not suspended on.
        suspend_after: Seconds. How long the server suggests polling the task
            before suspending (``suspendAfterMs``), or ``None`` when it sent
            none, or one that cannot be used.
    """

    job_key: str
    folder_key: str
    suspend_after: float | None = None


def _as_seconds(milliseconds: Any) -> float | None:
    """A whole, non-negative number of milliseconds as seconds, else ``None``.

    The same rule as ``McpTaskJob.suspend_after_ms`` in uipath-platform.
    """
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)):
        return None
    # An int is whole and finite whatever its size; only a float needs asking.
    if isinstance(milliseconds, float) and not milliseconds.is_integer():
        return None
    if milliseconds < 0:
        return None
    try:
        return milliseconds / 1000
    except OverflowError:
        # A JSON integer too large for a float: longer than any bound applied
        # to it afterwards, which is all that needs to be known about it.
        return math.inf


def _as_key(value: Any) -> str | None:
    """``value`` as a GUID string, or ``None`` when it is not one."""
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


def task_job(meta: Any) -> TaskJob | None:
    """Read the job annotation off a task's ``_meta``.

    The annotation comes from the server and ends up in a resume trigger and an
    Orchestrator request, so anything that is not exactly what is expected
    yields ``None``: a missing or malformed annotation costs the suspend, and
    the task is polled instead. It never fails the call. The same goes for an
    annotation at another version, or one that names a trigger other than a
    Job one.
    """
    # TODO: read this through ``McpTask.job`` once the uipath-platform floor is
    # 0.3.19, which parses the same annotation and applies the same version
    # and trigger-kind rule. Two things then stay here:
    # the folder is required (``McpTask.job`` names a job without one, which is
    # right for a reader but not for a caller about to suspend on it), and the
    # polling time is bounded (``polling_time_before_suspension``).
    if not isinstance(meta, Mapping):
        return None
    annotation = meta.get(UIPATH_TASK_META_KEY)
    if not isinstance(annotation, Mapping):
        return None

    # Read only at the version known here, and only when the trigger named is
    # a Job one. Anything else is a kind of resume not implemented here, so
    # the task is polled rather than suspended on by a guess.
    version = annotation.get("version")
    if type(version) is not int or version != UIPATH_TASK_ANNOTATION_VERSION:
        return None
    trigger = annotation.get("resumeTrigger")
    if (
        not isinstance(trigger, Mapping)
        or trigger.get("triggerType") != JOB_TRIGGER_TYPE
    ):
        return None

    job_key = _as_key(annotation.get("jobKey"))
    # Without the folder the resume would look for the job in the run's own
    # folder, and fault the run when it is somewhere else.
    folder_key = _as_key(annotation.get("folderKey"))
    if job_key is None or folder_key is None:
        return None

    # Only a hint: one that cannot be used is dropped and the job kept.
    return TaskJob(job_key, folder_key, _as_seconds(annotation.get("suspendAfterMs")))


def suspension_enabled() -> bool:
    """Whether a task call may suspend the run (default: yes).

    Set the ``DisableMcpTasksSuspension`` flag to ``true`` -- programmatically
    or via ``UIPATH_FEATURE_DisableMcpTasksSuspension`` -- to keep every task
    polled. Whether a task can be suspended on at all is the server's call: a
    server that does not serve ``com.uipath/tasks`` never names a job.
    """
    return not FeatureFlags.is_flag_enabled(SUSPENSION_KILL_SWITCH, default=False)


def _seconds_flag(name: str) -> float | None:
    """A duration flag in seconds, or ``None`` when unset or unusable."""
    value = FeatureFlags.get_flag(name)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring unparseable {name}={value!r}")
        return None
    # Zero is a real setting here ("suspend at once"); a negative one is not.
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def suspension_settings() -> SuspensionSettings:
    """The caller's own settings, from their flag; unset where it has none."""
    return SuspensionSettings(
        min_polling_time_before_suspension=_seconds_flag(MIN_POLLING_TIME_FLAG),
    )


def polling_time_before_suspension(settings: SuspensionSettings, job: TaskJob) -> float:
    """How long to poll a task, in seconds, before suspending on its job.

    The caller's own setting comes first, so that one run can be tuned without
    the server. Then the server's suggestion for this task, which it knows
    best. Then the default.

    The server's value is bounded by the poll budget: that is as long as a
    call is polled when it cannot suspend at all, so nothing the server sends
    can keep a run awake longer than it would have been without this module.
    """
    if settings.min_polling_time_before_suspension is not None:
        return settings.min_polling_time_before_suspension
    if job.suspend_after is not None:
        # Imported here: ``mcp_tasks`` imports this module.
        from .mcp_tasks import task_poll_budget_seconds

        return min(job.suspend_after, task_poll_budget_seconds())
    return DEFAULT_MIN_POLLING_TIME_BEFORE_SUSPENSION


def _scratchpad() -> Any:
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return None
    return configurable.get(CONFIG_KEY_SCRATCHPAD)


def can_suspend() -> bool:
    """Whether this call runs inside a graph node that can raise an interrupt."""
    return _scratchpad() is not None


def durable_call_in_play() -> bool:
    """Whether a call made now has to go through :func:`call_with_suspension`.

    It does where the run can suspend and suspension is on. It also does, with
    suspension switched off, when the node is being replayed with saved step
    values: one of them may be this call's, sent before the switch was thrown.
    Taking the plain path then would send ``tools/call`` a second time for
    work that is already running. On the durable path the saved task is read
    back instead, and the switch only keeps a new suspend from starting.
    """
    scratchpad = _scratchpad()
    if scratchpad is None:
        return False
    return suspension_enabled() or bool(getattr(scratchpad, "resume", None))


@dataclass(frozen=True)
class Submission(Generic[ResultT, HandleT]):
    """What sending the call produced: a result at once, or a task handle."""

    result: ResultT | None = None
    handle: HandleT | None = None


class TaskCall(Protocol[ResultT, HandleT]):
    """One tool call, as the suspend flow drives it.

    The flow is the same whichever client carries the call -- the agent tool
    layer's MCP session or the platform SDK's stateless ``McpService`` -- so
    the client is behind this protocol.
    """

    async def submit(self) -> Submission[ResultT, HandleT]:
        """Send ``tools/call``. Runs once per call, never on a replay."""

    async def wait(
        self, handle: HandleT, *, seconds: float | None, poll_first: bool
    ) -> ResultT:
        """Poll the task to its end.

        Args:
            handle: The task.
            seconds: How long to poll, or ``None`` for the client's own bound.
            poll_first: Ask for the task before the first sleep.

        Raises:
            PollWindowElapsed: ``seconds`` passed with the task still working.
        """

    async def peek(self, handle: HandleT) -> ResultT | None:
        """Ask for the task once: its result, or ``None`` while it works."""

    def dump_handle(self, handle: HandleT) -> dict[str, Any]:
        """The handle as plain JSON, to be saved in the checkpoint."""

    def load_handle(self, saved: dict[str, Any]) -> HandleT:
        """The handle back from what ``dump_handle`` saved."""

    def dump_result(self, result: ResultT) -> Any:
        """A result as plain JSON, to be saved in the checkpoint."""

    def load_result(self, saved: Any) -> ResultT:
        """The result back from what ``dump_result`` saved."""

    def job(self, handle: HandleT) -> TaskJob | None:
        """The job behind the task, when the server named one."""


@dataclass
class _Saved(SkipInterruptValue):
    """A step value kept in the checkpoint instead of suspending the run."""

    payload: dict[str, Any]

    @property
    def resume_value(self) -> dict[str, Any]:
        return {_PAYLOAD_KEY: self.payload}


def _read_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and isinstance(value.get(_PAYLOAD_KEY), dict):
        return value[_PAYLOAD_KEY]
    return None


def _wait_on_job(job: TaskJob) -> WaitJobRaw:
    """The interrupt that suspends the run until ``job`` ends.

    ``WaitJobRaw`` rather than ``WaitJob``: on resume the plain one reads the
    job's output and faults the run when the job did not succeed. Neither is
    wanted here -- the result is read with ``tasks/get``, which also reports a
    faulted or stopped job the way the tool layer expects -- so the resume only
    has to say the job has ended.
    """
    return WaitJobRaw(
        job=Job(key=job.job_key, folder_key=job.folder_key, id=0),
        process_folder_key=job.folder_key,
    )


async def call_with_suspension(
    call: TaskCall[ResultT, HandleT],
    *,
    tool_name: str,
    settings: SuspensionSettings | None = None,
    save_results: bool = True,
) -> ResultT:
    """Run one tool call, suspending the run while its task is long.

    Must run inside a graph node (see :func:`can_suspend`). A call the server
    answers at once and a task with no job named behave exactly as a polled
    call does.

    Args:
        call: The call and the client that carries it.
        tool_name: The tool, for log lines.
        settings: When to suspend; read from its flag when omitted.
        save_results: Save each result with the step that produced it, so that
            a node which suspends *later*, on something else, gets it back on
            the replay without the call being sent or the task polled again.
            A coded node needs that. A tool node runs one call and nothing
            after it, so it passes ``False`` and keeps results -- which can be
            large -- out of the checkpoint; only the task handle is saved.

    Returns:
        The tool's result.
    """
    settings = settings or suspension_settings()

    # Results of this pass that are not saved with their step (``save_results``).
    kept: list[ResultT] = []

    def finished(result: ResultT) -> _Saved:
        if save_results:
            return _Saved({"kind": "finished", "result": call.dump_result(result)})
        kept.append(result)
        return _Saved({"kind": "finished"})

    def not_resumable() -> AgentRuntimeError:
        # A step value that is not this call's, or a result that was not saved
        # with its step. Sending the call again here could run the tool twice,
        # so the call fails instead.
        return AgentRuntimeError(
            code=AgentRuntimeErrorCode.STATE_ERROR,
            title=f"MCP tool '{tool_name}' could not be resumed",
            detail=(
                f"The saved state of the call to MCP tool '{tool_name}' could not "
                "be read back on resume. This happens when the node calls "
                "interrupt() before the tool call; make the tool call first, or "
                "move the interrupt to a node of its own."
            ),
            category=UiPathErrorCategory.SYSTEM,
        )

    def result_of(step_value: Any) -> ResultT:
        payload = _read_payload(step_value)
        if payload is None or payload.get("kind") != "finished":
            raise not_resumable()
        if "result" in payload:
            return call.load_result(payload["result"])
        if kept:
            return kept[-1]
        raise not_resumable()

    @durable_interrupt
    async def submit() -> _Saved:
        # The polling time runs from when the call was sent, so the time the
        # server took to answer with a task counts towards it.
        submitted_at = time.time()
        submission = await call.submit()
        if submission.handle is None:
            return finished(submission.result)  # type: ignore[arg-type]
        return _Saved(
            {
                "kind": "task",
                "handle": call.dump_handle(submission.handle),
                "submitted_at": submitted_at,
            }
        )

    sent = await submit()
    submitted = _read_payload(sent)
    if submitted is None or submitted.get("kind") != "task":
        return result_of(sent)

    handle = call.load_handle(submitted["handle"])
    job = call.job(handle)
    resumed = False

    if job is not None:
        polling_ends_at = float(
            submitted["submitted_at"]
        ) + polling_time_before_suspension(settings, job)

        @durable_interrupt
        async def park() -> _Saved | WaitJobRaw:
            # Everything that comes before a suspend is part of this step, so a
            # replay after the resume skips it along with the suspend itself.
            if not suspension_enabled():
                # Switched off since the call was sent: no new suspend is
                # started, and the task is polled to its end.
                return finished(await call.wait(handle, seconds=None, poll_first=False))

            # The polling time runs from when the call was sent.
            remaining = polling_ends_at - time.time()
            if remaining > 0:
                try:
                    return finished(
                        await call.wait(handle, seconds=remaining, poll_first=False)
                    )
                except PollWindowElapsed:
                    # The poller stops when its next poll would not fit, which
                    # can be well before the time is up -- at once, for a poll
                    # interval longer than the polling time. The rest is waited
                    # out here.
                    remaining = polling_ends_at - time.time()
                    if remaining > 0:
                        await anyio.sleep(remaining)

            # One last look, so that a task which has finished by now costs a
            # poll and not a suspend.
            result = await call.peek(handle)
            if result is not None:
                return finished(result)
            logger.info(
                f"MCP task for tool '{tool_name}' is still working; suspending the "
                f"run until Orchestrator job {job.job_key} ends"
            )
            return _wait_on_job(job)

        parked = await park()
        if (_read_payload(parked) or {}).get("kind") == "finished":
            return result_of(parked)
        resumed = True

    @durable_interrupt
    async def collect() -> _Saved:
        # A step of its own, so that a node which suspends later on something
        # else gets this result back instead of polling the task again -- by
        # then the server may no longer have it.
        if resumed:
            # The job has ended, so the first poll is normally the last one.
            logger.info(f"Resumed for tool '{tool_name}'; reading its MCP task")
        return finished(await call.wait(handle, seconds=None, poll_first=resumed))

    return result_of(await collect())
