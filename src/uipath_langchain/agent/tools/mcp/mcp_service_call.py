"""Call a UiPath MCP tool from a coded agent, suspending the run on a long task.

A low-code agent reaches MCP tools through ``McpClient`` and its tool node. A
coded agent has no such node: it calls a tool from its own graph code, with the
platform SDK's ``McpService``. :func:`call_mcp_tool` is that call with the
suspend behaviour of ``mcp_task_suspension.py``::

    from uipath_langchain.agent.tools.mcp import call_mcp_tool

    async def reconcile(state: State) -> dict[str, Any]:
        result = await call_mcp_tool(
            "finance-tools", "reconcile_ledger", {"period": state["period"]}
        )
        return {"reconciliation": result["structuredContent"]}

The call is sent once however many times the node runs, and when the server
turns it into a long task the job suspends until the Orchestrator job behind
the task ends, then picks the result up with ``tasks/get``.
"""

import logging
from typing import Any

import httpx
from uipath.platform import UiPath
from uipath.platform.errors import EnrichedException
from uipath.platform.orchestrator.mcp_tasks import (
    McpTask,
    McpTaskStatus,
    McpToolCallError,
)

from .mcp_task_suspension import (
    PollWindowElapsed,
    Submission,
    SuspensionSettings,
    TaskJob,
    call_with_suspension,
    durable_call_in_play,
    task_job,
)
from .mcp_tasks import task_poll_budget_seconds

logger: logging.Logger = logging.getLogger(__name__)


# What one ``tasks/get`` can fail with. Whether a failure is a passing one is
# the SDK waiter's call (``wait_for_task_async``): it polls on through a
# transport error, a 5xx or a JSON-RPC internal error, and raises the rest.
_POLL_FAILURES = (httpx.TransportError, EnrichedException, McpToolCallError)


class _ServiceTaskCall:
    """One tool call over the platform SDK's stateless ``McpService``."""

    def __init__(
        self,
        sdk: Any,
        slug: str,
        tool_name: str,
        arguments: dict[str, Any] | None,
        folder_path: str | None,
    ) -> None:
        self._mcp = sdk.mcp
        self._slug = slug
        self._tool_name = tool_name
        self._arguments = arguments
        self._folder_path = folder_path

    async def submit(self) -> Submission[dict[str, Any], McpTask]:
        outcome = await self._mcp.call_tool_async(
            self._slug,
            self._tool_name,
            self._arguments,
            folder_path=self._folder_path,
        )
        if outcome.task is None:
            return Submission(result=outcome.result)
        return Submission(handle=outcome.task)

    async def wait(
        self, handle: McpTask, *, seconds: float | None, poll_first: bool
    ) -> dict[str, Any]:
        if poll_first:
            try:
                handle = await self._mcp.get_task_async(handle)
            except _POLL_FAILURES:
                # One failed read is no verdict on the task. The waiter asks
                # again on its interval, and it is the one that tells a
                # passing failure from a final one.
                logger.warning(
                    f"The first read of the MCP task for tool '{self._tool_name}' "
                    "failed; polling on",
                    exc_info=True,
                )
        try:
            return await self._mcp.wait_for_task_async(
                handle,
                timeout=task_poll_budget_seconds() if seconds is None else seconds,
            )
        except TimeoutError as elapsed:
            if seconds is None:
                raise
            raise PollWindowElapsed() from elapsed

    async def peek(self, handle: McpTask) -> dict[str, Any] | None:
        try:
            snapshot = await self._mcp.get_task_async(handle)
        except _POLL_FAILURES:
            # A look that failed says nothing about the task. The run suspends
            # on the job as it would have, and the read after the resume goes
            # through the waiter, which raises a failure that is final.
            logger.warning(
                f"Could not read the MCP task for tool '{self._tool_name}' "
                "before suspending",
                exc_info=True,
            )
            return None
        if snapshot.status == McpTaskStatus.WORKING or not isinstance(
            snapshot.status, McpTaskStatus
        ):
            return None
        # Settled: the waiter returns the result, or raises what the task
        # ended with, without polling again.
        return await self._mcp.wait_for_task_async(snapshot, timeout=0)

    def dump_handle(self, handle: McpTask) -> dict[str, Any]:
        # The endpoint is this client's own bookkeeping and is left out of a
        # dump on purpose (see McpTask.server_url), so it is saved beside it.
        return {
            "task": handle.model_dump(by_alias=True, mode="json", exclude_none=True),
            "server_url": handle.server_url,
        }

    def load_handle(self, saved: dict[str, Any]) -> McpTask:
        task = McpTask.model_validate(saved["task"])
        task.server_url = saved.get("server_url")
        return task

    def dump_result(self, result: dict[str, Any]) -> Any:
        return result

    def load_result(self, saved: Any) -> dict[str, Any]:
        return saved

    def job(self, handle: McpTask) -> TaskJob | None:
        return task_job((handle.model_extra or {}).get("_meta"))


async def call_mcp_tool(
    slug: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    folder_path: str | None = None,
    settings: SuspensionSettings | None = None,
) -> dict[str, Any]:
    """Call a tool on a UiPath MCP server and return its result.

    For a coded agent. Inside a LangGraph node the call is durable: it is sent
    once even if the node runs again, and a call the server turns into a long
    task suspends the job until the Orchestrator job behind the task ends,
    instead of keeping it running to poll. It is polled for
    ``minPollingTimeBeforeSuspension`` first, and suspends only if it is still
    working after that.

    Outside a graph node, or with suspension switched off, it is
    ``sdk.mcp.call_tool_and_wait_async``: the task is polled to its end.

    Because a suspended node is replayed from its first line on resume, call
    this before anything in the node that must not happen twice, or make that
    work a durable step of its own. In particular, do not call LangGraph's
    ``interrupt()`` earlier in the same node: the call cannot be resumed after
    it. Make this call first, or give the interrupt a node of its own.

    Args:
        slug: The MCP server's slug.
        tool_name: The tool to call.
        arguments: The tool's arguments.
        folder_path: The folder of the MCP server; defaults to the SDK's
            configured folder.
        settings: When to suspend; read from its flag when omitted.

    Returns:
        The tool's ``CallToolResult``, as a dict (``content``,
        ``structuredContent``, ``isError``).

    Raises:
        McpToolCallError: The server answered with a JSON-RPC error.
        McpTaskFailedError: The task failed.
        McpTaskCancelledError: The task was cancelled.
        McpTaskInputRequiredError: The task needs input this caller cannot give.
        TimeoutError: The task outlived the poll budget without the run being
            able to suspend on it.
    """
    sdk = UiPath()
    if not durable_call_in_play():
        return await sdk.mcp.call_tool_and_wait_async(
            slug,
            tool_name,
            arguments,
            folder_path=folder_path,
            timeout=task_poll_budget_seconds(),
        )

    return await call_with_suspension(
        _ServiceTaskCall(sdk, slug, tool_name, arguments, folder_path),
        tool_name=tool_name,
        settings=settings,
    )
