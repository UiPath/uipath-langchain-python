"""Tests for process_tool.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentProcessToolProperties,
    AgentProcessToolResourceConfig,
    AgentToolType,
)
from uipath.platform.common import WaitJob
from uipath.platform.orchestrator import Job

from uipath_langchain.agent.tools.process_tool import create_process_tool


@pytest.fixture
def process_resource():
    """Create a minimal process tool resource config."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.PROCESS,
        name="test_process",
        description="Test process description",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        properties=AgentProcessToolProperties(
            process_name="MyProcess",
            folder_path="/Shared/MyFolder",
        ),
    )


@pytest.fixture
def process_resource_with_inputs():
    """Create a process tool resource config with input arguments."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.PROCESS,
        name="data_processor",
        description="Process data with arguments",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
            },
        },
        properties=AgentProcessToolProperties(
            process_name="DataProcessor",
            folder_path="/Shared/DataFolder",
        ),
    )


class TestProcessToolMetadata:
    """Test that process tool has correct metadata for observability."""

    def test_process_tool_has_metadata(self, process_resource):
        """Test that process tool has metadata dict."""
        tool = create_process_tool(process_resource)

        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    def test_process_tool_metadata_has_tool_type(self, process_resource):
        """Test that metadata contains tool_type derived from resource type."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "process"

    def test_process_tool_metadata_has_display_name(self, process_resource):
        """Test that metadata contains display_name from process_name."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "MyProcess"

    def test_process_tool_metadata_has_folder_path(self, process_resource):
        """Test that metadata contains folder_path for span attributes."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"

    def test_process_tool_metadata_has_span_context(self, process_resource):
        """Test that metadata contains _span_context dict for tracing."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert "_span_context" in tool.metadata
        assert isinstance(tool.metadata["_span_context"], dict)

    def test_process_tool_metadata_tool_type_uses_resource_type(self):
        """Test that tool_type is derived from resource.type.lower()."""
        resource = AgentProcessToolResourceConfig(
            type=AgentToolType.PROCESS,
            name="test_process",
            description="Test",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentProcessToolProperties(
                process_name="MyProcess",
                folder_path="/Shared",
            ),
        )
        tool = create_process_tool(resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == resource.type.lower()


class TestProcessToolInvocation:
    """Test process tool invocation behavior: invoke then interrupt."""

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.process_tool.UiPath")
    async def test_invoke_calls_processes_invoke_async(
        self, mock_uipath_class, mock_interrupt, process_resource
    ):
        """Test that invoking the tool calls client.processes.invoke_async."""
        mock_job = MagicMock(spec=Job)
        mock_job.key = "job-key-123"
        mock_job.folder_key = "folder-key-123"

        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_uipath_class.return_value = mock_client

        mock_interrupt.return_value = {"output": "result"}

        tool = create_process_tool(process_resource)
        await tool.ainvoke({})

        mock_client.processes.invoke_async.assert_called_once_with(
            name="MyProcess",
            input_arguments={},
            folder_path="/Shared/MyFolder",
            attachments=[],
            parent_span_id=None,
            parent_operation_id=None,
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.process_tool.UiPath")
    async def test_invoke_interrupts_with_wait_job(
        self, mock_uipath_class, mock_interrupt, process_resource
    ):
        """Test that after invoking, the tool interrupts with WaitJob."""
        mock_job = MagicMock(spec=Job)
        mock_job.key = "job-key-456"
        mock_job.folder_key = "folder-key-456"

        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_uipath_class.return_value = mock_client

        mock_interrupt.return_value = {"output": "done"}

        tool = create_process_tool(process_resource)
        await tool.ainvoke({})

        mock_interrupt.assert_called_once()
        wait_job_arg = mock_interrupt.call_args[0][0]
        assert isinstance(wait_job_arg, WaitJob)
        assert wait_job_arg.job == mock_job
        assert wait_job_arg.process_folder_key == "folder-key-456"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.process_tool.UiPath")
    async def test_invoke_passes_input_arguments(
        self, mock_uipath_class, mock_interrupt, process_resource_with_inputs
    ):
        """Test that input arguments are correctly passed to invoke_async."""
        mock_job = MagicMock(spec=Job)
        mock_job.key = "job-key"
        mock_job.folder_key = "folder-key"

        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_uipath_class.return_value = mock_client

        mock_interrupt.return_value = {"result": "processed"}

        tool = create_process_tool(process_resource_with_inputs)
        await tool.ainvoke({"name": "test-data", "count": 42})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["input_arguments"] == {"name": "test-data", "count": 42}
        assert call_kwargs["name"] == "DataProcessor"
        assert call_kwargs["folder_path"] == "/Shared/DataFolder"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.process_tool.UiPath")
    async def test_invoke_returns_interrupt_value(
        self, mock_uipath_class, mock_interrupt, process_resource
    ):
        """Test that the tool returns the value from interrupt()."""
        mock_job = MagicMock(spec=Job)
        mock_job.key = "job-key"
        mock_job.folder_key = "folder-key"

        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_uipath_class.return_value = mock_client

        mock_interrupt.return_value = {"output_arg": "value123"}

        tool = create_process_tool(process_resource)
        result = await tool.ainvoke({})

        assert result == {"output_arg": "value123"}


class TestProcessToolSpanContext:
    """Test that _span_context is properly wired for tracing."""

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.process_tool.UiPath")
    async def test_span_context_parent_span_id_passed_to_invoke(
        self, mock_uipath_class, mock_interrupt, process_resource
    ):
        """Test that parent_span_id from _span_context is forwarded to invoke_async."""
        mock_job = MagicMock(spec=Job)
        mock_job.key = "job-key"
        mock_job.folder_key = "folder-key"

        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_uipath_class.return_value = mock_client

        mock_interrupt.return_value = {}

        tool = create_process_tool(process_resource)
        assert tool.metadata is not None

        # Simulate tracing setting parent_span_id via the shared _span_context
        tool.metadata["_span_context"]["parent_span_id"] = "span-abc-123"

        await tool.ainvoke({})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["parent_span_id"] == "span-abc-123"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.process_tool.UiPath")
    async def test_span_context_consumed_after_invoke(
        self, mock_uipath_class, mock_interrupt, process_resource
    ):
        """Test that parent_span_id is popped (consumed) from _span_context after use."""
        mock_job = MagicMock(spec=Job)
        mock_job.key = "job-key"
        mock_job.folder_key = "folder-key"

        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_uipath_class.return_value = mock_client

        mock_interrupt.return_value = {}

        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        tool.metadata["_span_context"]["parent_span_id"] = "span-xyz"

        await tool.ainvoke({})

        # parent_span_id should be consumed (popped) after invocation
        assert "parent_span_id" not in tool.metadata["_span_context"]

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt")
    @patch("uipath_langchain.agent.tools.process_tool.UiPath")
    async def test_span_context_defaults_to_none_when_empty(
        self, mock_uipath_class, mock_interrupt, process_resource
    ):
        """Test that parent_span_id defaults to None when _span_context is empty."""
        mock_job = MagicMock(spec=Job)
        mock_job.key = "job-key"
        mock_job.folder_key = "folder-key"

        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_uipath_class.return_value = mock_client

        mock_interrupt.return_value = {}

        tool = create_process_tool(process_resource)
        # Don't set any parent_span_id
        await tool.ainvoke({})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["parent_span_id"] is None
