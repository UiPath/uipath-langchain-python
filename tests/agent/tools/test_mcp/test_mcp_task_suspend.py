"""Tests for suspend-on-UiPath-task (PR-B).

A task-supporting MCP tool, when called against a UiPath MCP server, returns a
CreateTaskResult whose _meta marks it as a UiPath job. The tool then interrupts with a
WaitJobRaw (like process_tool), suspending the parent agent job until the child completes.
"""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import CreateTaskResult, Tool, ToolExecution
from uipath.agent.models.agent import AgentMcpTool, McpToolTaskSupport
from uipath.platform.common import WaitJobRaw

from uipath_langchain.agent.tools.mcp.mcp_client import McpClient
from uipath_langchain.agent.tools.mcp.mcp_tool import (
    _execution_from_server_tool,
    _is_task_augmentable,
    build_mcp_tool,
)


def _mcp_tool(task_support: str | None) -> AgentMcpTool:
    data: dict = {
        "name": "invoke-process",
        "description": "Run a process",
        "inputSchema": {"type": "object", "properties": {}},
    }
    if task_support is not None:
        data["execution"] = {"taskSupport": task_support}
    return AgentMcpTool.model_validate(data)


def _create_task_result(source: str = "orchestrator") -> CreateTaskResult:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return CreateTaskResult.model_validate(
        {
            "task": {
                "taskId": "job-key-1",
                "status": "working",
                "createdAt": now,
                "lastUpdatedAt": now,
                "ttl": 86_400_000,
            },
            "_meta": {
                "uipath.com/source": source,
                "uipath.com/jobKey": "job-key-1",
                "uipath.com/folderKey": "folder-key-1",
            },
        }
    )


class TestTaskAugmentableDetection:
    def test_optional_and_required_are_augmentable(self) -> None:
        assert _is_task_augmentable(_mcp_tool("optional")) is True
        assert _is_task_augmentable(_mcp_tool("required")) is True

    def test_forbidden_and_missing_are_not(self) -> None:
        assert _is_task_augmentable(_mcp_tool("forbidden")) is False
        assert _is_task_augmentable(_mcp_tool(None)) is False

    def test_execution_mapped_from_server_tool(self) -> None:
        tool = Tool(
            name="p",
            description="d",
            inputSchema={"type": "object", "properties": {}},
            execution=ToolExecution(taskSupport="optional"),
        )
        execution = _execution_from_server_tool(tool)
        assert execution is not None
        assert execution.task_support == McpToolTaskSupport.OPTIONAL

    def test_execution_none_when_server_tool_has_no_execution(self) -> None:
        tool = Tool(
            name="p",
            description="d",
            inputSchema={"type": "object", "properties": {}},
        )
        assert _execution_from_server_tool(tool) is None


class TestCallToolAsTask:
    async def test_sends_task_augmented_request(self) -> None:
        client = McpClient(config=MagicMock())
        session = MagicMock()
        create_result = _create_task_result()
        session.send_request = AsyncMock(return_value=create_result)
        client._ensure_session = AsyncMock(return_value=session)  # type: ignore[method-assign]
        client._client_initialized = True

        result = await client.call_tool_as_task("invoke-process", arguments={"a": 1})

        assert result is create_result
        sent_request = session.send_request.call_args[0][0]
        call_tool_request = sent_request.root
        assert call_tool_request.params.name == "invoke-process"
        assert call_tool_request.params.task is not None


class TestSuspendOnUiPathTask:
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_task_tool_starts_job_and_suspends_with_waitjob(
        self, mock_uipath: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        mcp_client = MagicMock()
        mcp_client.call_tool_as_task = AsyncMock(return_value=_create_task_result())

        resumed_job = MagicMock()
        resumed_job.state = "successful"
        mock_interrupt.return_value = resumed_job

        sdk = MagicMock()
        sdk.jobs.extract_output_async = AsyncMock(return_value='{"out": 1}')
        mock_uipath.return_value = sdk

        tool_fn = build_mcp_tool(_mcp_tool("optional"), mcp_client)
        result = await tool_fn(invoiceId="INV-1")

        mcp_client.call_tool_as_task.assert_awaited_once()
        # Suspended on a WaitJobRaw carrying the job + folder keys read from _meta.
        wait = mock_interrupt.call_args[0][0]
        assert isinstance(wait, WaitJobRaw)
        assert str(wait.job.key) == "job-key-1"
        assert str(wait.process_folder_key) == "folder-key-1"
        # Resume returns the child job's output.
        assert result == {"out": 1}

    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_non_task_tool_calls_synchronously(
        self, mock_interrupt: MagicMock
    ) -> None:
        mcp_client = MagicMock()
        sync_result = MagicMock()
        sync_result.content = "sync-result"
        mcp_client.call_tool = AsyncMock(return_value=sync_result)

        tool_fn = build_mcp_tool(_mcp_tool(None), mcp_client)
        result = await tool_fn(x=1)

        mcp_client.call_tool.assert_awaited_once()
        mock_interrupt.assert_not_called()
        assert result == "sync-result"
