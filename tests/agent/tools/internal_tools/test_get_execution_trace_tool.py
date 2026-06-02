"""Tests for get_execution_trace_tool.py module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from uipath.agent.models.agent import (
    AgentInternalGetExecutionTraceToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.internal_tools.get_execution_trace_tool import (
    PIMS_EXECUTION_TRACE_PATH,
    create_get_execution_trace_tool,
)


class TestCreateGetExecutionTraceTool:
    """Test cases for create_get_execution_trace_tool function."""

    @pytest.fixture
    def mock_llm(self):
        return AsyncMock()

    @pytest.fixture
    def resource_config(self):
        input_schema = {
            "type": "object",
            "properties": {
                "instanceId": {"type": "string"},
                "folderKey": {"type": "string"},
            },
            "required": ["instanceId"],
        }
        output_schema = {
            "type": "object",
            "properties": {
                "elementExecutions": {"type": "array"},
                "traceId": {"type": "string"},
            },
        }
        properties = AgentInternalGetExecutionTraceToolProperties(
            tool_type=AgentInternalToolType.GET_EXECUTION_TRACE
        )
        return AgentInternalToolResourceConfig(
            name="get_execution_trace",
            description="Fetch the audit trail (element executions) of a case instance by id.",
            input_schema=input_schema,
            output_schema=output_schema,
            properties=properties,
        )

    @staticmethod
    def _mock_uipath_response(mock_uipath_class, payload):
        response = Mock()
        response.json.return_value = payload
        mock_uipath = Mock()
        mock_uipath.api_client.request_async = AsyncMock(return_value=response)
        mock_uipath_class.return_value = mock_uipath
        return mock_uipath

    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath"
    )
    async def test_explicit_folder_key_sends_header(
        self, mock_uipath_class, resource_config, mock_llm
    ):
        mock_uipath = self._mock_uipath_response(
            mock_uipath_class,
            {"elementExecutions": [], "traceId": "trace-1"},
        )

        tool = create_get_execution_trace_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        result = await tool.coroutine(instanceId="abc-123", folderKey="folder-xyz")

        assert result == {"elementExecutions": [], "traceId": "trace-1"}
        mock_uipath.api_client.request_async.assert_awaited_once()
        args, kwargs = mock_uipath.api_client.request_async.call_args
        assert args == ("GET", PIMS_EXECUTION_TRACE_PATH.format(instance_id="abc-123"))
        assert kwargs == {"headers": {"x-uipath-folderkey": "folder-xyz"}}

    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath"
    )
    async def test_falls_back_to_runtime_folder_context(
        self, mock_uipath_class, resource_config, mock_llm
    ):
        mock_uipath = self._mock_uipath_response(
            mock_uipath_class,
            {"elementExecutions": [{"elementId": "e1"}], "traceId": "trace-2"},
        )

        tool = create_get_execution_trace_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        result = await tool.coroutine(instanceId="abc-123")

        assert result == {
            "elementExecutions": [{"elementId": "e1"}],
            "traceId": "trace-2",
        }
        args, kwargs = mock_uipath.api_client.request_async.call_args
        assert args == ("GET", PIMS_EXECUTION_TRACE_PATH.format(instance_id="abc-123"))
        assert kwargs == {"include_folder_headers": True}

    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath"
    )
    async def test_missing_instance_id_raises(
        self, mock_uipath_class, resource_config, mock_llm
    ):
        self._mock_uipath_response(mock_uipath_class, {})

        tool = create_get_execution_trace_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="instanceId"):
            await tool.coroutine(folderKey="folder-xyz")

    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath"
    )
    async def test_404_maps_to_agent_runtime_error(
        self,
        mock_uipath_class,
        resource_config,
        mock_llm,
        make_enriched_exception,
    ):
        mock_uipath = Mock()
        mock_uipath.api_client.request_async = AsyncMock(
            side_effect=make_enriched_exception(404, body="case not found")
        )
        mock_uipath_class.return_value = mock_uipath

        tool = create_get_execution_trace_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError):
            await tool.coroutine(instanceId="missing-id")

    def test_factory_registered_in_handlers(self):
        from uipath_langchain.agent.tools.internal_tools.internal_tool_factory import (
            _INTERNAL_TOOL_HANDLERS,
        )

        assert AgentInternalToolType.GET_EXECUTION_TRACE in _INTERNAL_TOOL_HANDLERS
        assert (
            _INTERNAL_TOOL_HANDLERS[AgentInternalToolType.GET_EXECUTION_TRACE]
            is create_get_execution_trace_tool
        )
