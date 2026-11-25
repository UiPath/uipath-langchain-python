"""Long-running A2A calls: send without blocking, then poll on timer wake-ups.

The message is sent once with ``return_immediately`` set. A task that is still
``submitted`` or ``working`` is polled in the job for a short while
(``initial_inline_wait``), so fast tasks never pay for a suspend. After that each
poll is a ``@durable_interrupt`` step that calls ``tasks/get``: a settled task ends
the loop, otherwise the step returns ``WaitUntil`` and the job suspends on an
Orchestrator Timer trigger, holding no robot slot until it fires. When the
message continues a task or a conversation, a poll that still shows the previous
turn is not taken as the reply (see ``_TurnCheck``).

On resume the node replays. Every step done earlier returns its saved value
without running its body: the send is never repeated, and only the poll after
the last saved one calls the remote agent. Each step yields exactly one
``interrupt()`` call on every run (one send, then one per poll), so the indexes
stay in step with LangGraph's interrupt counter. This relies on the runtime
resuming by interrupt id (``Command(resume={interrupt_id: value})``), as the
UiPath runtime does: a plain ``Command(resume=value)`` re-runs the woken poll
once more before saving it. Step bodies never raise for the
same reason: a failure is saved as a payload and raised after the step returns.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from logging import getLogger
from typing import Any

import a2a.utils.errors as a2a_errors
import httpx
from a2a.client import Client
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from a2a.types import (
    GetTaskRequest,
    Message,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskState,
)
from google.protobuf.json_format import MessageToDict, ParseDict
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph.config import get_config
from uipath.platform.common import WaitUntil

from uipath_langchain._utils.durable_interrupt import (
    SkipInterruptValue,
    durable_interrupt,
)

logger = getLogger(__name__)

_PAYLOAD_KEY = "__a2a_long_running__"

# States in which the remote agent is still working on its own. Every other
# state (terminal, input-required, auth-required) is the caller's turn.
_WAITING_STATES = frozenset(
    {TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING}
)

# How long after the send a continued task may still show its previous turn.
_PREVIOUS_TURN_WINDOW = timedelta(seconds=60)


@dataclass(frozen=True)
class A2aPollSettings:
    """How a long-running A2A call waits for the remote task.

    Attributes:
        initial_inline_wait: How long to poll inside the job before suspending.
        inline_poll_interval: Delay between polls inside the job.
        poll_intervals: Timer delays between suspended polls. The n-th suspend
            uses the n-th entry; the last one repeats.
        max_wait: How long after the send to keep waiting. After that the call
            returns the task as still working.
    """

    initial_inline_wait: timedelta = timedelta(seconds=60)
    inline_poll_interval: timedelta = timedelta(seconds=5)
    poll_intervals: tuple[timedelta, ...] = field(
        default=(
            timedelta(minutes=5),
            timedelta(minutes=10),
            timedelta(minutes=15),
            timedelta(minutes=30),
        )
    )
    max_wait: timedelta = timedelta(hours=24)

    def poll_interval(self, index: int) -> timedelta:
        """Timer delay before the poll that follows the ``index``-th suspend."""
        return self.poll_intervals[min(index, len(self.poll_intervals) - 1)]


DEFAULT_POLL_SETTINGS = A2aPollSettings()


@dataclass(frozen=True)
class A2aCallOutcome:
    """What a long-running A2A call ended with.

    Attributes:
        reply: The remote agent's message, or the last task seen.
        timed_out: The task was still running when ``max_wait`` passed, or the
            caller cannot suspend the run.
    """

    reply: Task | Message
    timed_out: bool = False


class A2aTaskLostError(a2a_errors.A2AError):
    """A poll no longer found the task the message started.

    Unlike a ``TaskNotFoundError`` from the send, the remote agent did receive
    the message, so sending it again may make it act twice.
    """


@dataclass
class _Saved(SkipInterruptValue):
    """A step result kept in the checkpoint instead of suspending the run."""

    payload: dict[str, Any]

    @property
    def resume_value(self) -> dict[str, Any]:
        return {_PAYLOAD_KEY: self.payload}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_waiting(task: Task) -> bool:
    """Whether the remote agent is still working on ``task`` on its own."""
    return task.HasField("status") and task.status.state in _WAITING_STATES


@dataclass(frozen=True)
class _TurnCheck:
    """Tells a continued task's previous turn apart from the reply to this message.

    A task that is continued (a UiPath conversational agent keeps one task per
    conversation) can briefly still return the previous turn's finished state,
    until the agent picks up the new message. That state is stamped before the
    send reply. The check needs the server to stamp its statuses, and it ends
    ``_PREVIOUS_TURN_WINDOW`` after the send, so a misread can delay the call but
    never stall it.

    Attributes:
        not_before: The send reply's status time, by the server's clock. ``None``
            when the message names no task or conversation, or the server does
            not stamp it.
        until: When the check ends, by the job's clock.
    """

    not_before: datetime | None
    until: datetime

    def shows_previous_turn(self, task: Task) -> bool:
        """Whether ``task`` is finished with a state set before the send."""
        if self.not_before is None or is_waiting(task) or _utcnow() >= self.until:
            return False
        stamp = _status_time(task)
        return stamp is not None and stamp < self.not_before


def _status_time(task: Task) -> datetime | None:
    if not (task.HasField("status") and task.status.HasField("timestamp")):
        return None
    return task.status.timestamp.ToDatetime(tzinfo=timezone.utc)


def _can_suspend() -> bool:
    """Whether this call runs inside a graph node that can raise an interrupt."""
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return False
    return configurable.get(CONFIG_KEY_SCRATCHPAD) is not None


def _is_transient(error: Exception) -> bool:
    """Whether a failed poll may succeed if retried on the next wake-up.

    Network errors, timeouts, 5xx, 408 and 429 are transient. Anything else
    fails the same way on every poll: an a2a protocol error, another HTTP status
    (401, 403, 404...), or an error the SDK does not wrap, such as a reply that
    does not parse.
    """
    if isinstance(error, (A2AClientTimeoutError, a2a_errors.InternalError)):
        return True
    if type(error) is not A2AClientError:
        return False
    cause = error.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        status = cause.response.status_code
        return status >= 500 or status in (408, 429)
    return isinstance(cause, httpx.RequestError)


def _permanent_poll_error(task_id: str, error: Exception) -> Exception | None:
    """The error a failed poll ends the call with, or ``None`` to retry the poll.

    A not-found becomes ``A2aTaskLostError``: the agent did receive the message,
    so callers must not take it for a not-found on the send.
    """
    if _is_transient(error):
        return None
    if isinstance(error, a2a_errors.TaskNotFoundError):
        return A2aTaskLostError(
            f"The agent received the message, but task {task_id} was no longer "
            f"found while waiting for its reply: {error}"
        )
    return error


def _log_reply(event: str, reply: Task | Message, started_at: datetime) -> None:
    elapsed = (_utcnow() - started_at).total_seconds()
    if isinstance(reply, Task):
        logger.info(
            "%s in %.1fs: task=%s context=%s state=%s",
            event,
            elapsed,
            reply.id,
            reply.context_id,
            TaskState.Name(reply.status.state),
        )
    else:
        logger.info("%s in %.1fs: message context=%s", event, elapsed, reply.context_id)


def _reply_payload(
    reply: Task | Message, *, started_at: datetime | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "task" if isinstance(reply, Task) else "message",
        "reply": MessageToDict(reply),
    }
    if started_at is not None:
        payload["started_at"] = started_at.isoformat()
    return payload


def _error_payload(error: Exception) -> dict[str, Any]:
    error_type = type(error).__name__
    if isinstance(error, a2a_errors.A2AError):
        return {"kind": "error", "error_type": error_type, "message": str(error)}
    return {
        "kind": "error",
        "error_type": "A2AError",
        "message": str(error) or error_type,
    }


def _read_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and isinstance(value.get(_PAYLOAD_KEY), dict):
        return value[_PAYLOAD_KEY]
    return None


def _load_reply(payload: dict[str, Any]) -> Task | Message:
    """Rebuild the saved reply, or raise the saved error as its a2a error type."""
    if payload["kind"] == "error":
        name = payload["error_type"]
        error_class = (
            A2aTaskLostError
            if name == A2aTaskLostError.__name__
            else getattr(a2a_errors, name, None)
        )
        if not (
            isinstance(error_class, type)
            and issubclass(error_class, a2a_errors.A2AError)
        ):
            error_class = a2a_errors.A2AError
        raise error_class(payload["message"])
    reply: Task | Message = Task() if payload["kind"] == "task" else Message()
    ParseDict(payload["reply"], reply, ignore_unknown_fields=True)
    return reply


async def _send_nonblocking(client: Client, message: Message) -> Task | Message:
    request = SendMessageRequest(
        message=message,
        configuration=SendMessageConfiguration(return_immediately=True),
    )
    async for response in client.send_message(request):
        if response.HasField("message"):
            return response.message
        if response.HasField("task"):
            return response.task
    raise a2a_errors.InvalidAgentResponseError("No response received.")


async def _wait_inline(
    client: Client, task: Task, settings: A2aPollSettings, check: _TurnCheck
) -> Task:
    """Poll ``task`` inside the job for up to ``initial_inline_wait``.

    A poll that still shows the previous turn is skipped: the last waiting state
    is kept, so that turn's reply is never returned for this message. A poll
    error a retry may fix leaves the task to the suspended polls; any other is
    raised, as the suspended polls would.
    """
    deadline = _utcnow() + settings.initial_inline_wait
    while is_waiting(task) and _utcnow() < deadline:
        await asyncio.sleep(settings.inline_poll_interval.total_seconds())
        try:
            polled = await client.get_task(GetTaskRequest(id=task.id))
        except Exception as e:
            logger.warning("A2A in-job poll of task %s failed", task.id, exc_info=True)
            permanent = _permanent_poll_error(task.id, e)
            if permanent is not None:
                raise permanent from e
            break
        if check.shows_previous_turn(polled):
            logger.info("A2A task %s still shows the previous turn", task.id)
            continue
        task = polled
    return task


async def send_and_wait(
    client: Client,
    message: Message,
    settings: A2aPollSettings | None = None,
) -> A2aCallOutcome:
    """Send ``message`` and wait for the task it starts to settle.

    Must run in the tool node when the task outlives ``initial_inline_wait``: the
    wait then suspends the run. Outside a graph node it returns the task as
    timed out instead.

    Raises:
        A2aTaskLostError: A poll no longer found the task. The agent did
            receive the message.
        A2AError: The send failed, or a poll failed in a way a retry would not
            fix (a protocol error, 401/403, a reply that does not parse...).
            Network errors, timeouts, 5xx, 408 and 429 are retried on the next
            wake-up. The a2a subclass (``TaskNotFoundError``,
            ``InvalidRequestError``...) is kept across replays.
    """
    settings = settings or DEFAULT_POLL_SETTINGS

    @durable_interrupt
    async def send() -> _Saved:
        started_at = _utcnow()
        logger.info(
            "A2A send: task=%s context=%s",
            message.task_id or "new",
            message.context_id or "new",
        )
        not_before: datetime | None = None
        try:
            reply = await _send_nonblocking(client, message)
            _log_reply("A2A send returned", reply, started_at)
            if isinstance(reply, Task):
                # A conversation restored from chat history is continued by its
                # context id alone, and AgentHub continues its task.
                if message.task_id or message.context_id:
                    not_before = _status_time(reply)
                check = _TurnCheck(not_before, started_at + _PREVIOUS_TURN_WINDOW)
                reply = await _wait_inline(client, reply, settings, check)
        except Exception as e:
            logger.warning("A2A send step failed", exc_info=True)
            return _Saved(_error_payload(e))
        payload = _reply_payload(reply, started_at=started_at)
        if not_before is not None:
            payload["not_before"] = not_before.isoformat()
        return _Saved(payload)

    sent = _read_payload(await send())
    if sent is None:
        raise a2a_errors.A2AError("A2A send step returned no saved reply")
    reply = _load_reply(sent)
    if not isinstance(reply, Task) or not is_waiting(reply):
        return A2aCallOutcome(reply)
    if not _can_suspend():
        return A2aCallOutcome(reply, timed_out=True)

    task_id = reply.id
    started_at = datetime.fromisoformat(sent["started_at"])
    deadline = started_at + settings.max_wait
    check = _TurnCheck(
        datetime.fromisoformat(sent["not_before"]) if "not_before" in sent else None,
        started_at + _PREVIOUS_TURN_WINDOW,
    )

    @durable_interrupt
    async def poll(index: int) -> _Saved | WaitUntil:
        now = _utcnow()
        try:
            task: Task | None = await client.get_task(GetTaskRequest(id=task_id))
        except Exception as e:
            permanent = _permanent_poll_error(task_id, e)
            if permanent is not None:
                logger.warning("A2A poll of task %s failed", task_id, exc_info=True)
                return _Saved(_error_payload(permanent))
            logger.warning(
                "A2A poll of task %s failed; retrying", task_id, exc_info=True
            )
            task = None
        if task is not None:
            _log_reply("A2A poll returned", task, now)
            if check.shows_previous_turn(task):
                logger.info("A2A task %s still shows the previous turn", task_id)
                task = None
        if task is not None and not is_waiting(task):
            return _Saved(_reply_payload(task))
        if now >= deadline:
            payload = _reply_payload(task or reply)
            payload["timed_out"] = True
            return _Saved(payload)
        resume_time = min(now + settings.poll_interval(index), deadline)
        logger.info(
            "A2A task %s still running; suspending until %s", task_id, resume_time
        )
        return WaitUntil(resume_time=resume_time)

    index = 0
    while True:
        polled = _read_payload(await poll(index))
        index += 1
        if polled is not None:
            return A2aCallOutcome(
                _load_reply(polled), timed_out=bool(polled.get("timed_out"))
            )
        # Woken by the timer (its resume value is ``{"resumeTime": ...}``): poll again.
