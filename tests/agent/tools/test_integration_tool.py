"""Tests for integration_tool.py module."""

import pytest
from uipath.agent.models.agent import (
    AgentIntegrationToolParameter,
    AgentIntegrationToolProperties,
    AgentIntegrationToolResourceConfig,
)
from uipath.platform.connections import ActivityParameterLocationInfo, Connection

from uipath_langchain.agent.tools.integration_tool import (
    convert_to_activity_metadata,
)


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
