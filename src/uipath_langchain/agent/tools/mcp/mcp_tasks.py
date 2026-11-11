"""MCP Tasks (io.modelcontextprotocol/tasks) support for the agent tool layer.

Implements the Tasks extension (ext-tasks, revision 2026-07-28) as an SDK
``ClientExtension`` with one ``ResultClaim``: a ``tools/call`` answered with
``resultType: "task"`` is claimed, polled to completion over the same session,
and resolved into the ordinary ``CallToolResult`` — invisible to agent
definitions (PRODEV-2012). The SDK activates the claim (and the capability
advertisement it implies) only on protocol versions that can deliver claimed
shapes, so legacy-era connections keep their wire byte-for-byte.

The models here mirror the ext-tasks wire (``ttlMs``/``pollIntervalMs``), not
the SDK's built-in ``Task`` type, which still speaks the superseded SEP-1686
field names (``ttl``/``pollInterval``).

Rollout model: the real per-tenant rollout control is the server-side
``McpTasks`` feature flag in AgentHub — a declaring client against a tenant
with the flag off gets plain synchronous behavior. The flags here are an
emergency opt-out (kill switch), not a rollout gate, because serverless
runtime images bake their package versions and cannot be rolled back per
tenant. They go through the SDK's shared ``FeatureFlags`` registry, so they
can be set programmatically by a host or injected as
``UIPATH_FEATURE_<name>`` — the same path gitops already uses to reach a
Python runtime (``EnablePythonGovernanceChecker``).
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
    CallToolResult,
    GetTaskRequest,
    GetTaskRequestParams,
    Result,
)
from pydantic import Field
from uipath.core.feature_flags import FeatureFlags

logger: logging.Logger = logging.getLogger(__name__)

TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"

#: Kill switch; set to ``false`` to restore the pre-Tasks behavior exactly.
TASKS_POLLING_FLAG = "McpTasksPolling"

#: Seconds one tool call may spend polling its task before giving up.
POLL_BUDGET_FLAG = "McpTasksPollBudgetSeconds"

# Leaves headroom inside the serverless runtime's 15-minute execution window:
# a tool that used to fail fast on HTTP timeout must not silently consume the
# agent's whole run polling. On budget exhaustion the task keeps running
# server-side and its id stays valid.
DEFAULT_POLL_BUDGET_SECONDS = 840.0

DEFAULT_POLL_INTERVAL_MS = 5000


def tasks_polling_enabled() -> bool:
    """Whether task-aware tool calls are enabled (default: enabled).

    Set the ``McpTasksPolling`` flag to ``false`` — programmatically or via
    ``UIPATH_FEATURE_McpTasksPolling`` — to disable the extension entirely,
    restoring the pre-Tasks behavior byte-for-byte.
    """
    return FeatureFlags.is_flag_enabled(TASKS_POLLING_FLAG, default=True)


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


def _interval_seconds(poll_interval_ms: int | None) -> float:
    interval = poll_interval_ms or DEFAULT_POLL_INTERVAL_MS
    return max(interval, 100) / 1000.0


async def _resolve_task(handle: McpTaskHandle, ctx: ClaimContext) -> CallToolResult:
    """Poll a claimed task to completion and return its final CallToolResult.

    Honors the server's ``pollIntervalMs`` (re-read on every snapshot) within
    the poll budget. The statuses are terminal per spec: ``completed`` carries
    the result, ``failed`` the JSON-RPC error, ``cancelled`` neither.

    Raises:
        MCPError: When the task reaches ``failed`` (the task's own error) or
            ``cancelled``, or reports ``input_required``, which the agent tool
            layer has no interactive party to satisfy.
        TimeoutError: When the poll budget elapses first. The task keeps
            running server-side; the message carries the task id.
    """
    deadline = time.monotonic() + task_poll_budget_seconds()
    interval = _interval_seconds(handle.poll_interval_ms)
    logger.info(
        f"Tool '{ctx.tool_name}' answered with task {handle.task_id}; "
        f"polling every {interval}s"
    )

    while True:
        if time.monotonic() + interval > deadline:
            raise TimeoutError(
                f"MCP task {handle.task_id} for tool '{ctx.tool_name}' did not "
                f"finish within the poll budget; it keeps running server-side."
            )
        await anyio.sleep(interval)

        snapshot = await ctx.session.send_request(
            GetTaskRequest(params=GetTaskRequestParams(task_id=handle.task_id)),
            McpTaskSnapshot,
        )

        if snapshot.status == "completed":
            return CallToolResult.model_validate(snapshot.result or {})

        if snapshot.status == "failed":
            error = snapshot.error or {}
            raise MCPError(
                error.get("code", INTERNAL_ERROR),
                error.get("message", "The MCP task failed."),
                error.get("data"),
            )

        if snapshot.status == "cancelled":
            raise MCPError(
                INTERNAL_ERROR,
                f"MCP task {handle.task_id} was cancelled before completing.",
            )

        if snapshot.status == "input_required":
            # The tool layer has no interactive party to answer an elicitation;
            # surfacing this beats polling the budget away on a parked task.
            raise MCPError(
                INTERNAL_ERROR,
                f"MCP task {handle.task_id} requires additional input, "
                "which the agent tool layer cannot provide.",
            )

        interval = _interval_seconds(snapshot.poll_interval_ms)


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
    )
    if not final.is_error:
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
    return {
        "extensions": {extension.identifier: extension.settings()},
        "result_claims": {extension.identifier: extension.claims()},
    }
