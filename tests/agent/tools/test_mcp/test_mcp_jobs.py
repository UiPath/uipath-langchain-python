"""Tests for the uipath.com/job MCP feature (PR-C1 client + PR-C2 LangGraph executor).

Covers: the initialize advertisement → ``is_job_aware``; the job-aware tool_fn
routing and the START/FETCH ``_meta`` the wrapper sends; and the
``LangGraphJobExecutor`` non-job / suspend-resume / faulted paths.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import InitializeResult
from uipath.platform.common import WaitJobRaw
from uipath.platform.mcp_jobs import JobStart, UiPathJobHandle
from uipath.platform.orchestrator import Job

from uipath_langchain.agent.tools.mcp import LangGraphJobExecutor, McpClient
from uipath_langchain.agent.tools.mcp.mcp_tool import build_mcp_tool

_INTERRUPT = "uipath_langchain._utils.durable_interrupt.decorator.interrupt"


def _init_result(meta: dict[str, Any] | None) -> InitializeResult:
    payload: dict[str, Any] = {
        "protocolVersion": "2025-11-25",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "test-server", "version": "1.0.0"},
    }
    if meta is not None:
        payload["_meta"] = meta
    return InitializeResult.model_validate(payload)


def _agent_mcp_tool() -> Any:
    from uipath.agent.models.agent import AgentMcpTool

    return AgentMcpTool(
        name="run_process",
        description="Runs a process",
        input_schema={"type": "object", "properties": {}},
    )


def _tool_result(*, meta: dict[str, Any] | None, content: Any) -> MagicMock:
    result = MagicMock()
    result.meta = meta
    result.content = content
    return result


# --- advertisement parsing (PR-C1) -----------------------------------------


def test_advertisement_marks_session_job_aware() -> None:
    client = McpClient(config=MagicMock())
    assert client.is_job_aware is False

    client._apply_job_advertisement(_init_result({"uipath.com/job": {"version": 1}}))

    assert client.is_job_aware is True
    assert client.job_version == 1


def test_no_advertisement_leaves_session_plain() -> None:
    client = McpClient(config=MagicMock())

    client._apply_job_advertisement(_init_result(None))

    assert client.is_job_aware is False
    assert client.job_version is None


# --- job-aware routing + START/FETCH _meta (PR-C1) -------------------------


class _CaptureExecutor:
    """Captures the start/fetch closures the wrapper hands the executor."""

    def __init__(self) -> None:
        self.start: Any = None
        self.fetch: Any = None
        self.tool_name: str | None = None

    async def run(self, *, start: Any, fetch: Any, tool_name: str) -> Any:
        self.start = start
        self.fetch = fetch
        self.tool_name = tool_name
        return "executor-ran"


def _job_aware_client(call_tool: AsyncMock) -> MagicMock:
    client = MagicMock(spec=McpClient)
    client.is_job_aware = True
    client.job_version = 1
    client.call_tool = call_tool
    return client


@pytest.mark.asyncio
async def test_job_aware_tool_delegates_to_executor() -> None:
    executor = _CaptureExecutor()
    client = _job_aware_client(AsyncMock())
    client.job_executor = executor

    tool_fn = build_mcp_tool(_agent_mcp_tool(), client)
    out = await tool_fn(city="here")

    assert out == "executor-ran"
    assert executor.tool_name == "run_process"


@pytest.mark.asyncio
async def test_start_sends_start_meta_and_parses_handle() -> None:
    call_tool = AsyncMock(
        return_value=_tool_result(
            meta={"uipath.com/job": {"key": "job-1", "folderKey": "folder-1"}},
            content="ignored",
        )
    )
    executor = _CaptureExecutor()
    client = _job_aware_client(call_tool)
    client.job_executor = executor

    tool_fn = build_mcp_tool(_agent_mcp_tool(), client)
    await tool_fn(city="here")
    start_outcome = await executor.start()

    assert start_outcome == JobStart(
        handle=UiPathJobHandle(job_key="job-1", folder_key="folder-1")
    )
    call_tool.assert_awaited_once_with(
        "run_process",
        arguments={"city": "here"},
        meta={"uipath.com/job": {"version": 1}},
    )


@pytest.mark.asyncio
async def test_start_without_handle_returns_normal_result() -> None:
    call_tool = AsyncMock(return_value=_tool_result(meta=None, content="plain output"))
    executor = _CaptureExecutor()
    client = _job_aware_client(call_tool)
    client.job_executor = executor

    tool_fn = build_mcp_tool(_agent_mcp_tool(), client)
    await tool_fn()
    start_outcome = await executor.start()

    assert start_outcome == JobStart(handle=None, result="plain output")


@pytest.mark.asyncio
async def test_fetch_sends_fetch_meta_with_no_arguments() -> None:
    call_tool = AsyncMock(return_value=_tool_result(meta=None, content="job result"))
    executor = _CaptureExecutor()
    client = _job_aware_client(call_tool)
    client.job_executor = executor

    tool_fn = build_mcp_tool(_agent_mcp_tool(), client)
    await tool_fn()
    fetched = await executor.fetch(
        UiPathJobHandle(job_key="job-1", folder_key="folder-1")
    )

    assert fetched == "job result"
    call_tool.assert_awaited_once_with(
        "run_process",
        arguments=None,
        meta={"uipath.com/job": {"key": "job-1", "folderKey": "folder-1"}},
    )


# --- LangGraphJobExecutor (PR-C2) ------------------------------------------


@pytest.mark.asyncio
async def test_executor_non_job_returns_result_without_interrupt() -> None:
    executor = LangGraphJobExecutor()

    async def start() -> JobStart:
        return JobStart(handle=None, result="plain")

    async def fetch(handle: UiPathJobHandle) -> Any:
        raise AssertionError("fetch must not run for a non-job result")

    with patch(_INTERRUPT) as mock_interrupt:
        out = await executor.run(start=start, fetch=fetch, tool_name="t")

    assert out == "plain"
    mock_interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_executor_suspends_with_waitjobraw_then_fetches_on_resume() -> None:
    executor = LangGraphJobExecutor()
    resumed_job = Job(id=0, key="job-1", state="successful", folder_key="folder-1")

    async def start() -> JobStart:
        return JobStart(handle=UiPathJobHandle(job_key="job-1", folder_key="folder-1"))

    async def fetch(handle: UiPathJobHandle) -> Any:
        return {"fetched": handle.job_key, "folder": handle.folder_key}

    # Patch interrupt to simulate suspend→resume: it returns the terminal Job.
    with patch(_INTERRUPT, return_value=resumed_job) as mock_interrupt:
        out = await executor.run(start=start, fetch=fetch, tool_name="t")

    assert out == {"fetched": "job-1", "folder": "folder-1"}
    assert mock_interrupt.call_count == 1
    wait_value = mock_interrupt.call_args.args[0]
    assert isinstance(wait_value, WaitJobRaw)
    assert wait_value.job.key == "job-1"
    assert wait_value.process_folder_key == "folder-1"


@pytest.mark.asyncio
async def test_executor_surfaces_faulted_job_without_fetching() -> None:
    executor = LangGraphJobExecutor()
    faulted_job = Job(
        id=0, key="job-1", state="faulted", folder_key="folder-1", info="it broke"
    )

    async def start() -> JobStart:
        return JobStart(handle=UiPathJobHandle(job_key="job-1", folder_key="folder-1"))

    async def fetch(handle: UiPathJobHandle) -> Any:
        raise AssertionError("fetch must not run for a faulted job")

    with patch(_INTERRUPT, return_value=faulted_job):
        out = await executor.run(start=start, fetch=fetch, tool_name="t")

    assert out == "it broke"
