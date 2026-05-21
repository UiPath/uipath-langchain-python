"""Tests for process_tool.py."""

import pytest
from uipath.agent.models.agent import (
    AgentProcessToolProperties,
    AgentProcessToolResourceConfig,
    AgentToolType,
)
from uipath.platform.common import WaitJob

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
def flow_resource():
    """Create a process tool resource config of type Flow (Maestro Flow release)."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.FLOW,
        name="test_flow",
        description="Test flow description",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        properties=AgentProcessToolProperties(
            process_name="MyFlow",
            folder_path="/Shared/Flows",
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

    @pytest.mark.parametrize(
        "resource_fixture,expected_tool_type",
        [
            ("process_resource", "process"),
            ("flow_resource", "flow"),
        ],
    )
    def test_metadata_tool_type_derived_from_resource_type(
        self, resource_fixture, expected_tool_type, request
    ):
        """tool_type metadata is derived from resource.type.lower()."""
        resource = request.getfixturevalue(resource_fixture)
        tool = create_process_tool(resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == expected_tool_type
        assert tool.metadata["tool_type"] == resource.type.lower()

    @pytest.mark.parametrize(
        "resource_fixture,expected_display_name",
        [
            ("process_resource", "MyProcess"),
            ("flow_resource", "MyFlow"),
        ],
    )
    def test_metadata_display_name_from_process_name(
        self, resource_fixture, expected_display_name, request
    ):
        """display_name metadata is taken from properties.process_name."""
        resource = request.getfixturevalue(resource_fixture)
        tool = create_process_tool(resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == expected_display_name

    def test_process_tool_metadata_has_folder_path(self, process_resource, monkeypatch):
        """Test that metadata contains folder_path for span attributes."""
        monkeypatch.setenv("UIPATH_FOLDER_PATH", "/Shared/TestFolder")
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/TestFolder"

    def test_process_tool_metadata_has_span_context(self, process_resource):
        """Test that metadata contains _span_context dict for tracing."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        assert "_span_context" in tool.metadata
        assert isinstance(tool.metadata["_span_context"], dict)


class TestProcessToolInvocation:
    """Test process tool invocation behavior: invoke then interrupt."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resource_fixture,expected_name,expected_folder",
        [
            ("process_resource", "MyProcess", "/Shared/MyFolder"),
            ("flow_resource", "MyFlow", "/Shared/Flows"),
        ],
    )
    async def test_invoke_calls_processes_invoke_async(
        self,
        resource_fixture,
        expected_name,
        expected_folder,
        mock_process_invocation,
        monkeypatch,
        request,
    ):
        """Both Process and Flow tools invoke client.processes.invoke_async."""
        monkeypatch.setenv("UIPATH_FOLDER_PATH", expected_folder)
        mock_client, _, _, _ = mock_process_invocation
        resource = request.getfixturevalue(resource_fixture)

        tool = create_process_tool(resource)
        await tool.ainvoke({})

        mock_client.processes.invoke_async.assert_called_once_with(
            name=expected_name,
            input_arguments={},
            folder_path=expected_folder,
            attachments=[],
            parent_span_id=None,
            parent_operation_id=None,
            run_as_me=None,
        )

    @pytest.mark.asyncio
    async def test_invoke_interrupts_with_wait_job(
        self, process_resource, mock_process_invocation
    ):
        """Test that after invoking, the tool interrupts with WaitJobRaw."""
        _, mock_interrupt, mock_job, _ = mock_process_invocation

        tool = create_process_tool(process_resource)
        await tool.ainvoke({})

        mock_interrupt.assert_called_once()
        wait_job_arg = mock_interrupt.call_args[0][0]
        assert isinstance(wait_job_arg, WaitJob)
        assert wait_job_arg.job == mock_job
        assert wait_job_arg.process_folder_key == mock_job.folder_key

    @pytest.mark.asyncio
    async def test_invoke_passes_input_arguments(
        self,
        process_resource_with_inputs,
        mock_process_invocation,
        monkeypatch,
    ):
        """Test that input arguments are correctly passed to invoke_async."""
        monkeypatch.setenv("UIPATH_FOLDER_PATH", "/Shared/DataFolder")
        mock_client, _, _, _ = mock_process_invocation

        tool = create_process_tool(process_resource_with_inputs)
        await tool.ainvoke({"name": "test-data", "count": 42})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["input_arguments"] == {"name": "test-data", "count": 42}
        assert call_kwargs["name"] == "DataProcessor"
        assert call_kwargs["folder_path"] == "/Shared/DataFolder"

    @pytest.mark.asyncio
    async def test_invoke_returns_output_from_extract(
        self, process_resource, mock_process_invocation
    ):
        """Test that the tool returns the extracted job output on success."""
        mock_client, _, _, _ = mock_process_invocation
        mock_client.jobs.extract_output_async.return_value = (
            '{"output_arg": "value123"}'
        )

        tool = create_process_tool(process_resource)
        result = await tool.ainvoke({})

        assert result == {"output_arg": "value123"}

    @pytest.mark.asyncio
    async def test_invoke_returns_error_message_on_faulted_job(
        self, process_resource, mock_process_invocation
    ):
        """Test that the tool returns an error message string when the job is faulted."""
        _, _, _, mock_resumed_job = mock_process_invocation
        mock_resumed_job.state = "faulted"
        mock_resumed_job.job_error = None
        mock_resumed_job.info = "Something went wrong in the workflow"

        tool = create_process_tool(process_resource)
        result = await tool.ainvoke({})

        assert isinstance(result, str)
        assert "Something went wrong in the workflow" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resource_fixture",
        ["process_resource", "flow_resource"],
    )
    async def test_bts_key_is_non_agent_variant(
        self, resource_fixture, mock_process_invocation, request
    ):
        """Non-agent process tools (Process, Flow) store the BTS key as 'wait_for_job_key'."""
        _, _, mock_job, _ = mock_process_invocation
        resource = request.getfixturevalue(resource_fixture)

        tool = create_process_tool(resource)
        assert tool.metadata is not None

        await tool.ainvoke({})

        bts_context = tool.metadata["_bts_context"]
        assert bts_context.get("wait_for_job_key") == mock_job.key
        assert "wait_for_agent_job_key" not in bts_context


class TestProcessToolSpanContext:
    """Test that _span_context is properly wired for tracing."""

    @pytest.mark.asyncio
    async def test_span_context_parent_span_id_passed_to_invoke(
        self, process_resource, mock_process_invocation
    ):
        """Test that parent_span_id from _span_context is forwarded to invoke_async."""
        mock_client, _, _, _ = mock_process_invocation

        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        tool.metadata["_span_context"]["parent_span_id"] = "span-abc-123"

        await tool.ainvoke({})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["parent_span_id"] == "span-abc-123"

    @pytest.mark.asyncio
    async def test_span_context_consumed_after_invoke(
        self, process_resource, mock_process_invocation
    ):
        """Test that parent_span_id is popped (consumed) from _span_context after use."""
        tool = create_process_tool(process_resource)
        assert tool.metadata is not None
        tool.metadata["_span_context"]["parent_span_id"] = "span-xyz"

        await tool.ainvoke({})

        assert "parent_span_id" not in tool.metadata["_span_context"]

    @pytest.mark.asyncio
    async def test_span_context_defaults_to_none_when_empty(
        self, process_resource, mock_process_invocation
    ):
        """Test that parent_span_id defaults to None when _span_context is empty."""
        mock_client, _, _, _ = mock_process_invocation

        tool = create_process_tool(process_resource)
        await tool.ainvoke({})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["parent_span_id"] is None


class TestProcessToolRunAsMe:
    """Test RunAsMe propagation passed top-down from tool factory."""

    @pytest.mark.asyncio
    async def test_run_as_me_true_passed_to_invoke(
        self, process_resource, mock_process_invocation
    ):
        """Test RunAsMe=True is forwarded to invoke_async when set."""
        mock_client, _, _, _ = mock_process_invocation

        tool = create_process_tool(process_resource, run_as_me=True)
        await tool.ainvoke({})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["run_as_me"] is True

    @pytest.mark.asyncio
    async def test_run_as_me_false_sends_none(
        self, process_resource, mock_process_invocation
    ):
        """Test RunAsMe=None when run_as_me=False (default)."""
        mock_client, _, _, _ = mock_process_invocation

        tool = create_process_tool(process_resource, run_as_me=False)
        await tool.ainvoke({})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["run_as_me"] is None

    @pytest.mark.asyncio
    async def test_run_as_me_default_sends_none(
        self, process_resource, mock_process_invocation
    ):
        """Test RunAsMe=None when run_as_me not specified (default)."""
        mock_client, _, _, _ = mock_process_invocation

        tool = create_process_tool(process_resource)
        await tool.ainvoke({})

        call_kwargs = mock_client.processes.invoke_async.call_args[1]
        assert call_kwargs["run_as_me"] is None
