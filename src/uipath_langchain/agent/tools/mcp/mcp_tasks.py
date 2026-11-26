"""MCP Tasks (io.modelcontextprotocol/tasks) support for the agent tool layer.

Implements the Tasks extension (ext-tasks, revision 2026-07-28) as an SDK
``ClientExtension`` with one ``ResultClaim``: a ``tools/call`` answered with
``resultType: "task"`` is claimed, polled to completion over the same session,
and resolved into the ordinary ``CallToolResult`` — invisible to agent
definitions. The SDK activates the claim (and the capability
advertisement it implies) only on protocol versions that can deliver claimed
shapes, so legacy-era connections keep their wire byte-for-byte.

The models here mirror the ext-tasks wire (``ttlMs``/``pollIntervalMs``), not
the SDK's built-in ``Task`` type, which still speaks the superseded SEP-1686
field names (``ttl``/``pollInterval``).

Rollout model: whether tasks happen at all is the server's call, not this
client's. A server that does not advertise the extension can never answer with
a task, so a client that declares it simply keeps the plain synchronous
behavior. The flags here are an emergency opt-out (a kill switch, phrased
negatively like the package's others) rather than a rollout gate, because a
deployed runtime image carries a fixed package version. They go through the
SDK's shared ``FeatureFlags`` registry, so a host can set them
programmatically or inject them as ``UIPATH_FEATURE_<name>``.
"""

import logging
import time
from typing import Any, Literal

import anyio
from mcp import ClientSession
from mcp.client.extension import (
    ClaimContext,
    ClientExtension,
    ResultClaim,
    UnexpectedClaimedResult,
)
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INTERNAL_ERROR,
    REQUEST_TIMEOUT,
    CallToolResult,
    GetTaskRequest,
    GetTaskRequestParams,
    Result,
)
from pydantic import Field, ValidationError
from uipath.core.feature_flags import FeatureFlags

from .mcp_task_suspension import (
    UIPATH_TASKS_CAPABILITY,
    UIPATH_TASKS_EXTENSION_ID,
    suspension_enabled,
)

logger: logging.Logger = logging.getLogger(__name__)

TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"

#: Kill switch; set it to ``true`` to restore the pre-Tasks behavior exactly.
TASKS_POLLING_KILL_SWITCH = "DisableMcpTasksPolling"

#: Seconds one tool call may spend polling its task before giving up.
POLL_BUDGET_FLAG = "McpTasksPollBudgetSeconds"

# Fourteen minutes: long enough for a genuinely long tool call, short enough to
# leave room inside a typical agent execution window. It bounds ONE call, not
# the run -- a run whose first task starts late, or which makes several
# task-shaped calls, can still reach the host's limit with the budget intact,
# and the host ends it there. The point is narrower: a tool that used to fail
# fast on an HTTP timeout must not silently spend the whole run polling. A
# deployment that wants a tighter bound sets the flag below.
DEFAULT_POLL_BUDGET_SECONDS = 840.0

DEFAULT_POLL_INTERVAL_MS = 5000

# Ceiling on one tasks/get, regardless of how much budget is left. Bounding a
# poll by the whole remaining budget made a stalled connection cost the entire
# call: with the default budget, one black-holed socket sat there for fourteen
# minutes and then reported a timeout, while the same stall under a short
# budget was noticed in seconds and re-polled. The loop is the retry, so one
# attempt only needs to outlast a healthy server's answer.
MAX_POLL_REQUEST_SECONDS = 30.0

# Consecutive failed polls absorbed before a resolve gives up. The loop is the
# only retry a poll gets (the client's session-recovery retry deliberately does
# not cover resolution, which would resubmit the tool call), so this keeps a
# server that is answering nothing but errors from eating the whole budget and
# then reporting a timeout instead of the error.
MAX_POLL_FAILURES = 5


def tasks_polling_enabled() -> bool:
    """Whether task-aware tool calls are enabled (default: enabled).

    Set the ``DisableMcpTasksPolling`` flag to ``true`` — programmatically or
    via ``UIPATH_FEATURE_DisableMcpTasksPolling`` — to disable the extension
    entirely, restoring the pre-Tasks behavior byte-for-byte. The flag is
    phrased negatively to match the other kill switches in this package
    (``DisableBatchTransformFromAttachment``, ``DisableDeepRagFromAttachments``).
    """
    return not FeatureFlags.is_flag_enabled(TASKS_POLLING_KILL_SWITCH, default=False)


def task_poll_budget_seconds() -> float:
    """The maximum seconds a single tool call may spend polling its task.

    Reads the ``McpTasksPollBudgetSeconds`` flag, falling back to
    ``DEFAULT_POLL_BUDGET_SECONDS`` when unset, unparseable, or non-positive.
    """
    value = FeatureFlags.get_flag(POLL_BUDGET_FLAG)
    if value is None:
        return DEFAULT_POLL_BUDGET_SECONDS

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring unparseable {POLL_BUDGET_FLAG}={value!r}; "
            f"using the default of {DEFAULT_POLL_BUDGET_SECONDS}s"
        )
        return DEFAULT_POLL_BUDGET_SECONDS

    return parsed if parsed > 0 else DEFAULT_POLL_BUDGET_SECONDS


class McpTaskHandle(Result):
    """The ``resultType: "task"`` answer to a ``tools/call`` (flat, per ext-tasks)."""

    result_type: Literal["task"] = Field("task", alias="resultType")
    task_id: str = Field(alias="taskId")
    status: str
    status_message: str | None = Field(None, alias="statusMessage")
    poll_interval_ms: int | None = Field(None, alias="pollIntervalMs")


class McpTaskSnapshot(Result):
    """One ``tasks/get`` answer; ``result``/``error`` are set on terminal states."""

    task_id: str = Field(alias="taskId")
    status: str
    status_message: str | None = Field(None, alias="statusMessage")
    poll_interval_ms: int | None = Field(None, alias="pollIntervalMs")
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def _task_ref(task_id: str) -> str:
    """A task id shortened for a message or a log line.

    An MCP task id is a self-contained handle, not a short key: a UiPath server
    mints one of 368 characters, and the first 27 of those are identical across
    tasks (a header on the encrypted blob). Taking a prefix therefore
    distinguishes nothing -- every task reads the same -- so this takes both
    ends. Short ids, which is what tests use, are left alone.

    Whoever can reach the server can read or cancel a task with its id, and a
    message or a log line travels further than the call that produced it; on
    the agent path this text is also read by a model, where a few hundred
    characters of opaque handle is noise it may well repeat. The full id stays
    in the one log line written when the task is first seen, which is what an
    operator needs to find a task that outlived its wait.
    """
    if len(task_id) <= 32:
        return task_id
    return f"{task_id[:12]}...{task_id[-8:]}"


def _interval_seconds(poll_interval_ms: int | None) -> float:
    interval = poll_interval_ms or DEFAULT_POLL_INTERVAL_MS
    return max(interval, 100) / 1000.0


class McpTaskPollBudgetExhausted(MCPError):
    """The poll budget ran out with the task still working.

    Still an ``MCPError`` with ``REQUEST_TIMEOUT``, which is what the tool layer
    maps. The type of its own is for a caller that set the budget on purpose
    and has something else to do when it runs out -- suspend the run -- and
    must not take one stalled poll's ``REQUEST_TIMEOUT`` for that.
    """


def _budget_exhausted(task_id: str, tool_name: str) -> McpTaskPollBudgetExhausted:
    """The one budget-exhaustion error, naming the task that outlived the budget.

    An ``MCPError`` rather than a ``TimeoutError`` because this is the headline
    new failure mode of the whole feature, and the tool layer maps exactly one
    exception type: ``mcp_tool.tool_fn`` catches ``MCPError`` and turns it into
    a categorized ``AgentRuntimeError``. A bare ``TimeoutError`` would walk
    straight past that and out of the tool as an uncategorized exception.

    The task is deliberately **not** cancelled: it is a real Orchestrator job
    doing real work, and the agent giving up on waiting is no reason to abort
    it mid-flight. The cost is that its result is dropped, and an LLM that
    retries the tool starts a second job -- which is still the lesser harm. The
    task is identified in the message by a shortened reference; the full id is
    in the log line written when the task was first seen, which is what an
    operator needs to find the one left running.
    """
    return McpTaskPollBudgetExhausted(
        REQUEST_TIMEOUT,
        f"MCP task {_task_ref(task_id)} for tool '{tool_name}' did not finish "
        f"within the poll budget; it keeps running server-side.",
    )


def _tolerate_poll_failure(
    handle: McpTaskHandle, ctx: ClaimContext, failure: MCPError, failures: int
) -> int:
    """Count a transient poll failure, or re-raise it.

    ``INTERNAL_ERROR`` is what a gateway's non-JSON 5xx or 429 becomes on the
    client transport, so the loop absorbs it: the next tick re-polls the same
    task id inside the same deadline. ``REQUEST_TIMEOUT`` is absorbed too while
    the budget holds -- one poll hitting its per-request ceiling says the
    connection stalled, not that the task is gone. Anything else is either the
    server's answer about *this* task -- an unknown id, a terminated session --
    or a transport this resolver cannot rebuild, and has to reach the caller.

    Returns:
        int: The new consecutive-failure count.

    Raises:
        MCPError: The failure itself, when it is not transient or the cap is
            reached.
    """
    failures += 1
    if (
        failure.code not in (INTERNAL_ERROR, REQUEST_TIMEOUT)
        or failures >= MAX_POLL_FAILURES
    ):
        raise failure

    logger.warning(
        f"Transient failure polling MCP task {_task_ref(handle.task_id)} for tool "
        f"'{ctx.tool_name}' ({failures}/{MAX_POLL_FAILURES}): {failure}"
    )
    return failures


async def _resolve_task(
    handle: McpTaskHandle,
    ctx: ClaimContext,
    *,
    budget_seconds: float | None = None,
    poll_first: bool = False,
) -> CallToolResult:
    """Poll a claimed task to completion and return its final CallToolResult.

    Honors the server's ``pollIntervalMs`` (re-read on every snapshot) within
    the poll budget. The statuses are terminal per spec: ``completed`` carries
    the result, ``failed`` the JSON-RPC error, ``cancelled`` neither.

    Args:
        handle: The task.
        ctx: The session and tool the task belongs to.
        budget_seconds: How long to poll. Defaults to the poll budget flag.
        poll_first: Ask for the task before the first sleep. For a caller that
            has reason to think the task is already over, such as one resumed
            because the job behind it ended.

    Raises:
        MCPError: When the task reaches ``failed`` (the task's own error) or
            ``cancelled``, reports ``input_required``, which the agent tool
            layer has no interactive party to satisfy, or outlives the poll
            budget (``REQUEST_TIMEOUT``; the task keeps running server-side).
            Every failure out of here is an ``MCPError`` so that the tool
            layer's single ``except MCPError`` can categorize it.
    """
    deadline = time.monotonic() + (
        task_poll_budget_seconds() if budget_seconds is None else budget_seconds
    )
    interval = _interval_seconds(handle.poll_interval_ms)
    failures = 0
    logger.info(
        f"Tool '{ctx.tool_name}' answered with task {handle.task_id}; "
        f"polling every {interval}s"
    )

    sleep = 0.0 if poll_first else interval
    while True:
        if time.monotonic() + sleep >= deadline:
            # No room left for both the wait and the poll that would follow it.
            raise _budget_exhausted(handle.task_id, ctx.tool_name)
        await anyio.sleep(sleep)
        # Only the first request of a `poll_first` call goes out at once. A
        # failed one is retried on the normal interval, not in a tight loop.
        sleep = interval

        # Re-read the clock after sleeping, and bound the request by what is
        # left: a sleep clamped to the deadline followed by an unbounded
        # tasks/get would let a stalled server answer long after the budget
        # expired -- and answer `completed`, which the caller would accept.
        # Capped as well, so one stalled connection costs a poll rather than
        # the whole call.
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise _budget_exhausted(handle.task_id, ctx.tool_name)

        try:
            snapshot = await ctx.session.send_request(
                GetTaskRequest(params=GetTaskRequestParams(task_id=handle.task_id)),
                McpTaskSnapshot,
                request_read_timeout_seconds=min(budget, MAX_POLL_REQUEST_SECONDS),
            )
        except MCPError as poll_error:
            if time.monotonic() >= deadline:
                # Out of budget: whatever the poll failed with, the call is
                # over and the caller gets the one budget-exhaustion error.
                raise _budget_exhausted(handle.task_id, ctx.tool_name) from poll_error
            failures = _tolerate_poll_failure(handle, ctx, poll_error, failures)
            continue

        failures = 0

        settled = _settled_result(handle, ctx, snapshot)
        if settled is not None:
            return settled

        interval = sleep = _interval_seconds(snapshot.poll_interval_ms)


def _settled_result(
    handle: McpTaskHandle, ctx: ClaimContext, snapshot: McpTaskSnapshot
) -> CallToolResult | None:
    """What a snapshot settles the call with, or ``None`` while the task works.

    Raises:
        MCPError: The task ``failed``, was ``cancelled``, or is parked on
            ``input_required``.
    """
    if snapshot.status == "completed":
        return _completed_result(handle, ctx, snapshot)

    if snapshot.status == "failed":
        error = snapshot.error or {}
        # The task's own code is reported in the message rather than
        # re-raised as this error's code. The tool layer reads the code to
        # decide whether the *connection* died
        # (`McpClient.is_session_error`), so a task failing with, say,
        # -32000 would otherwise be announced to the user as "the
        # connection to the server was terminated".
        code = error.get("code", INTERNAL_ERROR)
        message = error.get("message", "The MCP task failed.")
        raise MCPError(
            INTERNAL_ERROR,
            f"MCP task {_task_ref(handle.task_id)} failed: {message} (code {code})",
            error.get("data"),
        )

    if snapshot.status == "cancelled":
        raise MCPError(
            INTERNAL_ERROR,
            f"MCP task {_task_ref(handle.task_id)} was cancelled before completing.",
        )

    if snapshot.status == "input_required":
        # The tool layer has no interactive party to answer an elicitation;
        # surfacing this beats polling the budget away on a parked task.
        raise MCPError(
            INTERNAL_ERROR,
            f"MCP task {_task_ref(handle.task_id)} requires additional input, "
            "which the agent tool layer cannot provide.",
        )

    return None


async def _peek_task(handle: McpTaskHandle, ctx: ClaimContext) -> CallToolResult | None:
    """Ask for the task once: its result, or ``None`` while it is still working.

    A look that fails for a reason a later poll may not (the errors the poll
    loop absorbs) also answers ``None``: it says nothing about the task.

    Raises:
        MCPError: The task settled with a failure, or the look failed in a way
            no later poll would fix (an unknown task id, a dead transport).
    """
    try:
        snapshot = await ctx.session.send_request(
            GetTaskRequest(params=GetTaskRequestParams(task_id=handle.task_id)),
            McpTaskSnapshot,
            request_read_timeout_seconds=MAX_POLL_REQUEST_SECONDS,
        )
    except MCPError as poll_error:
        if poll_error.code in (INTERNAL_ERROR, REQUEST_TIMEOUT):
            logger.warning(
                f"Transient failure reading MCP task {_task_ref(handle.task_id)} "
                f"for tool '{ctx.tool_name}': {poll_error}"
            )
            return None
        raise
    return _settled_result(handle, ctx, snapshot)


def _completed_result(
    handle: McpTaskHandle, ctx: ClaimContext, snapshot: McpTaskSnapshot
) -> CallToolResult:
    """The completed task's payload, as the ``CallToolResult`` the caller expects.

    A ``completed`` task that carries nothing usable is still a failure of this
    tool call, and it has to arrive as an ``MCPError``: ``CallToolResult``
    requires ``content``, so validating ``{}`` raises a ``ValidationError``,
    which is not what the tool layer catches -- it would leave the tool as an
    uncategorized exception rather than a mapped tool failure.
    """
    try:
        return CallToolResult.model_validate(snapshot.result or {})
    except ValidationError as invalid:
        raise MCPError(
            INTERNAL_ERROR,
            f"MCP task {_task_ref(handle.task_id)} for tool '{ctx.tool_name}' completed "
            f"without a usable result: {invalid}",
        ) from invalid


class McpTasksExtension(ClientExtension):
    """Declares the Tasks extension and claims its ``resultType: "task"`` shape.

    The canonical declaration, usable directly with the high-level
    ``Client(extensions=[...])``. ``McpClient`` drives the low-level
    ``ClientSession``, which takes the same information as two mappings —
    see :func:`session_extension_kwargs`.
    """

    identifier = TASKS_EXTENSION_ID

    def claims(self) -> tuple[ResultClaim[McpTaskHandle], ...]:
        """The single claim: task-shaped tools/call answers, polled to completion."""
        return (
            ResultClaim(
                result_type="task",
                model=McpTaskHandle,
                resolve=_resolve_task,
            ),
        )


async def resolve_claimed_result(
    session: ClientSession,
    tool_name: str,
    result: CallToolResult | Result,
    *,
    budget_seconds: float | None = None,
    poll_first: bool = False,
) -> CallToolResult:
    """Finish a claimed ``tools/call`` answer, passing an ordinary one through.

    The low-level ``ClientSession`` parses a claimed shape but leaves resolving
    it to the host — the high-level ``Client`` does this in ``call_tool``, and
    ``McpClient`` drives the session directly, so it does the same here:
    resolve the claim, then revalidate the output schema exactly as the
    non-claimed path does (never for an ``isError`` result).
    """
    if isinstance(result, CallToolResult):
        return result

    if not isinstance(result, McpTaskHandle):
        # No other claim is registered, so this cannot happen through our
        # session; surfacing it beats returning a shape the caller cannot read.
        raise UnexpectedClaimedResult(result)

    final = await _resolve_task(
        result,
        ClaimContext(session=session, tool_name=tool_name, read_timeout_seconds=None),
        budget_seconds=budget_seconds,
        poll_first=poll_first,
    )
    if not final.is_error:
        await session.validate_tool_result(tool_name, final)
    return final


async def peek_claimed_task(
    session: ClientSession, tool_name: str, handle: McpTaskHandle
) -> CallToolResult | None:
    """One ``tasks/get``: the task's validated result, or ``None`` while it works."""
    final = await _peek_task(
        handle,
        ClaimContext(session=session, tool_name=tool_name, read_timeout_seconds=None),
    )
    if final is not None and not final.is_error:
        await session.validate_tool_result(tool_name, final)
    return final


def session_extension_kwargs() -> dict[str, Any]:
    """``ClientSession`` kwargs declaring Tasks, or nothing when opted out.

    The low-level session takes the advertisement (``extensions``) and the
    claimed shapes (``result_claims``) as separate mappings, both keyed by
    extension identifier. The SDK activates them only on protocol versions
    that can deliver claimed shapes, so a legacy-era connection neither
    advertises the extension nor can receive a task.
    """
    if not tasks_polling_enabled():
        return {}

    extension = McpTasksExtension()
    extensions: dict[str, dict[str, Any]] = {extension.identifier: extension.settings()}
    if suspension_enabled():
        # The UiPath name of the same protocol. It adds no result shape of its
        # own -- the task it answers with is the one claimed above -- so it is
        # declared with the trigger kinds a run can suspend on, and nothing
        # more. A server that serves it names the job behind the task.
        extensions[UIPATH_TASKS_EXTENSION_ID] = dict(UIPATH_TASKS_CAPABILITY)
    return {
        "extensions": extensions,
        "result_claims": {extension.identifier: extension.claims()},
    }
