"""Tests for get_case_state_tool.py module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from uipath.agent.models.agent import (
    AgentInternalGetCaseStateToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.internal_tools.get_case_state_tool import (
    PIMS_INSTANCE_PATH,
    create_get_case_state_tool,
)


class TestCreateGetCaseStateTool:
    """Test cases for create_get_case_state_tool function."""

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
            "properties": {"state": {"type": "string"}},
            "required": ["state"],
        }
        properties = AgentInternalGetCaseStateToolProperties(
            tool_type=AgentInternalToolType.GET_CASE_STATE
        )
        return AgentInternalToolResourceConfig(
            name="get_case_state",
            description="Fetch the current state of a case by instance id.",
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
    @patch("uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath")
    async def test_explicit_folder_key_sends_header(
        self, mock_uipath_class, resource_config, mock_llm
    ):
        mock_uipath = self._mock_uipath_response(
            mock_uipath_class, {"state": "Resolved"}
        )

        tool = create_get_case_state_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        result = await tool.coroutine(instanceId="abc-123", folderKey="folder-xyz")

        assert result == {"state": "Resolved"}
        mock_uipath.api_client.request_async.assert_awaited_once()
        args, kwargs = mock_uipath.api_client.request_async.call_args
        assert args == ("GET", PIMS_INSTANCE_PATH.format(instance_id="abc-123"))
        assert kwargs == {"headers": {"x-uipath-folderkey": "folder-xyz"}}

    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch("uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath")
    async def test_falls_back_to_runtime_folder_context(
        self, mock_uipath_class, resource_config, mock_llm
    ):
        mock_uipath = self._mock_uipath_response(
            mock_uipath_class, {"state": "InProgress"}
        )

        tool = create_get_case_state_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        result = await tool.coroutine(instanceId="abc-123")

        assert result == {"state": "InProgress"}
        args, kwargs = mock_uipath.api_client.request_async.call_args
        assert args == ("GET", PIMS_INSTANCE_PATH.format(instance_id="abc-123"))
        assert kwargs == {"include_folder_headers": True}

    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch("uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath")
    async def test_missing_instance_id_raises(
        self, mock_uipath_class, resource_config, mock_llm
    ):
        self._mock_uipath_response(mock_uipath_class, {})

        tool = create_get_case_state_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        with pytest.raises(ValueError, match="instanceId"):
            await tool.coroutine(folderKey="folder-xyz")

    @patch(
        "uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.mockable",
        lambda **kwargs: lambda f: f,
    )
    @patch("uipath_langchain.agent.tools.internal_tools._pims_read_tool_factory.UiPath")
    async def test_404_maps_to_agent_runtime_error(
        self,
        mock_uipath_class,
        resource_config,
        mock_llm,
        make_enriched_exception,
    ):
        mock_uipath = Mock()
        mock_uipath.api_client.request_async = AsyncMock(
            side_effect=make_enriched_exception(404, body="instance not found")
        )
        mock_uipath_class.return_value = mock_uipath

        tool = create_get_case_state_tool(resource_config, mock_llm)
        assert tool.coroutine is not None
        with pytest.raises(AgentRuntimeError):
            await tool.coroutine(instanceId="missing-id")

    def test_factory_registered_in_handlers(self):
        from uipath_langchain.agent.tools.internal_tools.internal_tool_factory import (
            _INTERNAL_TOOL_HANDLERS,
        )

        assert AgentInternalToolType.GET_CASE_STATE in _INTERNAL_TOOL_HANDLERS
        assert (
            _INTERNAL_TOOL_HANDLERS[AgentInternalToolType.GET_CASE_STATE]
            is create_get_case_state_tool
        )
