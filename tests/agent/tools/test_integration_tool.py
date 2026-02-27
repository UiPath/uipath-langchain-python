"""Tests for integration_tool.py module."""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolMessage
from uipath.agent.models.agent import (
    AgentIntegrationToolParameter,
    AgentIntegrationToolProperties,
    AgentIntegrationToolResourceConfig,
)
from uipath.platform.connections import ActivityParameterLocationInfo, Connection
from uipath.platform.errors import EnrichedException

from uipath_langchain.agent.tools.integration_tool import (
    convert_to_activity_metadata,
    create_integration_tool,
)
from uipath_langchain.agent.tools.tool_node import UiPathToolNode


class TestConvertToIntegrationServiceMetadata:
    """Test cases for convert_to_activity_metadata function."""

    @pytest.fixture
    def common_connection(self):
        """Common connection object used by all tests."""
        return Connection(
            id="test-connection-id", name="Test Connection", element_instance_id=12345
        )

    @pytest.fixture
    def base_properties_factory(self, common_connection):
        """Factory for creating base properties with common connection."""

        def _create_properties(
            method="POST",
            tool_path="/api/test",
            object_name="test_object",
            tool_display_name="Test Tool",
            tool_description="Test tool description",
            parameters=None,
        ):
            return AgentIntegrationToolProperties(
                method=method,
                tool_path=tool_path,
                object_name=object_name,
                tool_display_name=tool_display_name,
                tool_description=tool_description,
                connection=common_connection,
                parameters=parameters or [],
            )

        return _create_properties

    @pytest.fixture
    def resource_factory(self, base_properties_factory):
        """Factory for creating resource config with reusable properties."""

        def _create_resource(
            name="test_tool",
            description="Test tool",
            properties=None,
            **properties_kwargs,
        ):
            if properties is None:
                properties = base_properties_factory(**properties_kwargs)

            return AgentIntegrationToolResourceConfig(
                name=name,
                description=description,
                properties=properties,
                input_schema={},
            )

        return _create_resource

    def test_basic_conversion(self, resource_factory):
        """Test basic conversion with minimal parameters."""
        param = AgentIntegrationToolParameter(
            name="test_param", type="string", field_location="body"
        )
        resource = resource_factory(parameters=[param])

        result = convert_to_activity_metadata(resource)

        assert result.object_path == "/api/test"
        assert result.method_name == "POST"
        assert result.content_type == "application/json"
        assert isinstance(result.parameter_location_info, ActivityParameterLocationInfo)

    def test_getbyid_method_normalization(self, resource_factory):
        """Test that GETBYID method is normalized to GET."""
        resource = resource_factory(method="GETBYID")

        result = convert_to_activity_metadata(resource)

        assert result.method_name == "GET"

    def test_jsonpath_parameter_handling_nested_field(self, resource_factory):
        """Test handling of jsonpath parameter names with nested fields should extract top-level field only."""
        param = AgentIntegrationToolParameter(
            name="metadata.field.test", type="string", field_location="body"
        )
        resource = resource_factory(
            name="create_tool",
            description="Create tool",
            tool_path="/api/create",
            object_name="create_object",
            tool_display_name="Create Tool",
            tool_description="Create tool description",
            parameters=[param],
        )

        result = convert_to_activity_metadata(resource)

        # DESIRED BEHAVIOR: Should extract only the top-level field "metadata"
        assert "metadata" in result.parameter_location_info.body_fields
        assert len(result.parameter_location_info.body_fields) == 1

    @pytest.mark.parametrize(
        "param_name,expected_field",
        [
            ("attachments[*]", "attachments"),
            ("attachments[0]", "attachments"),
            ("attachments[1]", "attachments"),
            ("attachments[10]", "attachments"),
            ("attachments[*][*]", "attachments"),
            ("attachments[*][*][*]", "attachments"),
            ("attachments[*][0][*]", "attachments"),
            ("attachments[*].property", "attachments"),
        ],
    )
    def test_jsonpath_parameter_handling_array_notation(
        self, resource_factory, param_name, expected_field
    ):
        """Test handling of jsonpath parameter names with array notation should extract top-level field only."""
        param = AgentIntegrationToolParameter(
            name=param_name, type="string", field_location="body"
        )
        resource = resource_factory(
            name="create_tool",
            description="Create tool",
            tool_path="/api/create",
            object_name="create_object",
            tool_display_name="Create Tool",
            tool_description="Create tool description",
            parameters=[param],
        )

        result = convert_to_activity_metadata(resource)

        # DESIRED BEHAVIOR: Should extract only the top-level field
        assert expected_field in result.parameter_location_info.body_fields
        assert param_name not in result.parameter_location_info.body_fields
        assert len(result.parameter_location_info.body_fields) == 1

    def test_jsonpath_parameter_handling_multiple_nested_same_root(
        self, resource_factory
    ):
        """Test that multiple parameters with same root field are consolidated into one top-level field."""
        params = [
            AgentIntegrationToolParameter(
                name="metadata.field1", type="string", field_location="body"
            ),
            AgentIntegrationToolParameter(
                name="metadata.field2", type="string", field_location="body"
            ),
            AgentIntegrationToolParameter(
                name="metadata.nested.field", type="string", field_location="body"
            ),
        ]
        resource = resource_factory(
            name="create_tool",
            description="Create tool",
            tool_path="/api/create",
            object_name="create_object",
            tool_display_name="Create Tool",
            tool_description="Create tool description",
            parameters=params,
        )

        result = convert_to_activity_metadata(resource)

        # DESIRED BEHAVIOR: Should have only "metadata" once in body_fields
        assert "metadata" in result.parameter_location_info.body_fields
        assert len(result.parameter_location_info.body_fields) == 1
        # These should NOT be present
        assert "metadata.field1" not in result.parameter_location_info.body_fields
        assert "metadata.field2" not in result.parameter_location_info.body_fields
        assert "metadata.nested.field" not in result.parameter_location_info.body_fields

    def test_json_body_section_from_body_structure(self, resource_factory):
        """Test that jsonBodySection is extracted from body_structure."""
        param = AgentIntegrationToolParameter(
            name="prompt", type="string", field_location="body"
        )
        resource = resource_factory(parameters=[param])
        resource.properties.body_structure = {
            "contentType": "multipart",
            "jsonBodySection": "RagRequest",
        }

        result = convert_to_activity_metadata(resource)

        assert result.content_type == "multipart/form-data"
        assert result.json_body_section == "RagRequest"

    def test_json_body_section_none_when_not_specified(self, resource_factory):
        """Test that json_body_section is None when bodyStructure has no jsonBodySection."""
        param = AgentIntegrationToolParameter(
            name="prompt", type="string", field_location="body"
        )
        resource = resource_factory(parameters=[param])
        resource.properties.body_structure = {"contentType": "multipart"}

        result = convert_to_activity_metadata(resource)

        assert result.content_type == "multipart/form-data"
        assert result.json_body_section is None

    def test_json_body_section_none_when_no_body_structure(self, resource_factory):
        """Test that json_body_section is None when body_structure is None."""
        param = AgentIntegrationToolParameter(
            name="prompt", type="string", field_location="body"
        )
        resource = resource_factory(parameters=[param])

        result = convert_to_activity_metadata(resource)

        assert result.content_type == "application/json"
        assert result.json_body_section is None

    def test_parameter_location_mapping_simple_fields(self, resource_factory):
        """Test parameter mapping for simple field names across different locations."""
        params = [
            AgentIntegrationToolParameter(
                name="id", type="string", field_location="path"
            ),
            AgentIntegrationToolParameter(
                name="search", type="string", field_location="query"
            ),
            AgentIntegrationToolParameter(
                name="authorization", type="string", field_location="header"
            ),
            AgentIntegrationToolParameter(
                name="user", type="string", field_location="body"
            ),
        ]
        resource = resource_factory(
            name="update_user_tool",
            description="Update user tool",
            tool_path="/api/users/{id}",
            object_name="user_object",
            tool_display_name="Update User Tool",
            tool_description="Update user tool description",
            parameters=params,
        )

        result = convert_to_activity_metadata(resource)

        # Simple field names should be added as-is for non-body locations
        assert "id" in result.parameter_location_info.path_params
        assert len(result.parameter_location_info.path_params) == 1

        assert "search" in result.parameter_location_info.query_params
        assert len(result.parameter_location_info.query_params) == 1

        assert "authorization" in result.parameter_location_info.header_params
        assert len(result.parameter_location_info.header_params) == 1

        assert "user" in result.parameter_location_info.body_fields
        assert len(result.parameter_location_info.body_fields) == 1


def _make_enriched_exception(
    status_code: int, body: str = "error"
) -> EnrichedException:
    """Create an EnrichedException from a mock HTTPStatusError."""
    request = httpx.Request("POST", "https://example.com/elements_/v3/test")
    response = httpx.Response(
        status_code=status_code,
        content=body.encode("utf-8"),
        request=request,
    )
    http_error = httpx.HTTPStatusError(
        message=f"{status_code} Error",
        request=request,
        response=response,
    )
    return EnrichedException(http_error)


def _make_integration_resource(
    connection_id: str = "test-conn-id",
) -> AgentIntegrationToolResourceConfig:
    """Create a minimal integration tool resource config for testing."""
    connection = Connection(
        id=connection_id, name="Test Connection", element_instance_id=12345
    )
    properties = AgentIntegrationToolProperties(
        method="POST",
        tool_path="/api/query",
        object_name="query_object",
        tool_display_name="Run KQL Query",
        tool_description="Runs a KQL query",
        connection=connection,
        parameters=[
            AgentIntegrationToolParameter(
                name="query", type="string", field_location="body"
            ),
        ],
    )
    return AgentIntegrationToolResourceConfig(
        name="Run_KQL_Query",
        description="Runs a KQL query against the data source",
        properties=properties,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


class TestIntegrationToolWrapperErrorHandling:
    """Test cases for HTTP 4xx error handling in integration_tool_wrapper.

    Verifies that HTTP 4xx errors from Integration Service are returned
    to the LLM as error ToolMessages instead of crashing the agent.
    """

    @pytest.fixture
    def resource(self) -> AgentIntegrationToolResourceConfig:
        return _make_integration_resource()

    @pytest.fixture
    def mock_state(self) -> Any:
        """State with a single AI message containing a tool call."""
        tool_call = {
            "name": "Run_KQL_Query",
            "args": {"query": "test query"},
            "id": "call_abc",
        }
        ai_message = AIMessage(content="Running query", tool_calls=[tool_call])

        class _State:
            messages = [ai_message]

        return _State()

    @patch("uipath_langchain.agent.tools.integration_tool.UiPath")
    async def test_http_400_returned_as_error_tool_message(
        self,
        mock_uipath_cls: Any,
        resource: AgentIntegrationToolResourceConfig,
        mock_state: Any,
    ) -> None:
        """HTTP 400 Bad Request should be caught and returned as error ToolMessage."""
        mock_sdk = mock_uipath_cls.return_value
        mock_sdk.connections.invoke_activity_async = AsyncMock(
            side_effect=_make_enriched_exception(400, "Bad Request: invalid KQL syntax")
        )

        tool = create_integration_tool(resource)

        node = UiPathToolNode(
            tool,
            awrapper=tool.awrapper,  # type: ignore[attr-defined]
        )
        result = await node._afunc(mock_state)

        assert result is not None
        assert "messages" in result
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert msg.status == "error"
        assert "400" in msg.content
        assert "Bad Request: invalid KQL syntax" in msg.content
        assert msg.name == "Run_KQL_Query"
        assert msg.tool_call_id == "call_abc"

    @patch("uipath_langchain.agent.tools.integration_tool.UiPath")
    async def test_http_404_returned_as_error_tool_message(
        self,
        mock_uipath_cls: Any,
        resource: AgentIntegrationToolResourceConfig,
        mock_state: Any,
    ) -> None:
        """HTTP 404 Not Found should be caught and returned as error ToolMessage."""
        mock_sdk = mock_uipath_cls.return_value
        mock_sdk.connections.invoke_activity_async = AsyncMock(
            side_effect=_make_enriched_exception(404, "Resource not found")
        )

        tool = create_integration_tool(resource)

        node = UiPathToolNode(
            tool,
            awrapper=tool.awrapper,  # type: ignore[attr-defined]
        )
        result = await node._afunc(mock_state)

        assert result is not None
        assert "messages" in result
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert msg.status == "error"
        assert "404" in msg.content
        assert "Resource not found" in msg.content

    @patch("uipath_langchain.agent.tools.integration_tool.UiPath")
    async def test_http_500_still_propagates(
        self,
        mock_uipath_cls: Any,
        resource: AgentIntegrationToolResourceConfig,
        mock_state: Any,
    ) -> None:
        """HTTP 500 Internal Server Error should still propagate as an exception."""
        mock_sdk = mock_uipath_cls.return_value
        mock_sdk.connections.invoke_activity_async = AsyncMock(
            side_effect=_make_enriched_exception(500, "Internal Server Error")
        )

        tool = create_integration_tool(resource)

        node = UiPathToolNode(
            tool,
            awrapper=tool.awrapper,  # type: ignore[attr-defined]
        )

        with pytest.raises(EnrichedException) as exc_info:
            await node._afunc(mock_state)

        assert exc_info.value.status_code == 500

    @patch("uipath_langchain.agent.tools.integration_tool.UiPath")
    async def test_successful_call_returns_result(
        self,
        mock_uipath_cls: Any,
        resource: AgentIntegrationToolResourceConfig,
        mock_state: Any,
    ) -> None:
        """Successful tool invocations should return the result unchanged."""
        expected_result = {"rows": [{"col1": "val1"}]}
        mock_sdk = mock_uipath_cls.return_value
        mock_sdk.connections.invoke_activity_async = AsyncMock(
            return_value=expected_result
        )

        tool = create_integration_tool(resource)

        node = UiPathToolNode(
            tool,
            awrapper=tool.awrapper,  # type: ignore[attr-defined]
        )
        result = await node._afunc(mock_state)

        assert result is not None
        assert "messages" in result
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "rows" in msg.content
        assert "col1" in msg.content
        assert msg.name == "Run_KQL_Query"
        assert msg.tool_call_id == "call_abc"
