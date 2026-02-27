"""Tests for integration_tool.py module."""

from unittest.mock import MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentIntegrationToolParameter,
    AgentIntegrationToolProperties,
    AgentIntegrationToolResourceConfig,
    AgentToolArgumentArgumentProperties,
    AgentToolStaticArgumentProperties,
)
from uipath.platform.connections import ActivityParameterLocationInfo, Connection

from uipath_langchain.agent.exceptions import AgentStartupError
from uipath_langchain.agent.tools.integration_tool import (
    _is_param_name_to_jsonpath,
    _param_name_to_segments,
    convert_integration_parameters_to_argument_properties,
    convert_to_activity_metadata,
    create_integration_tool,
    strip_template_enums_from_schema,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
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


class TestConvertIntegrationParametersToArgumentProperties:
    """Test cases for convert_integration_parameters_to_argument_properties function."""

    def test_static_parameter_converted(self):
        """Static fieldVariant converts to AgentToolStaticArgumentProperties."""
        params = [
            AgentIntegrationToolParameter(
                name="api_key",
                type="string",
                value="my-secret-key",
                field_location="header",
                field_variant="static",
            ),
        ]

        result = convert_integration_parameters_to_argument_properties(params)

        assert "$['api_key']" in result
        prop = result["$['api_key']"]
        assert isinstance(prop, AgentToolStaticArgumentProperties)
        assert prop.is_sensitive is False
        assert prop.value == "my-secret-key"

    def test_argument_parameter_converted(self):
        """Argument fieldVariant converts to AgentToolArgumentArgumentProperties with argument_path extracted from {{...}}."""
        params = [
            AgentIntegrationToolParameter(
                name="user_id",
                type="string",
                value="{{userId}}",
                field_location="path",
                field_variant="argument",
            ),
        ]

        result = convert_integration_parameters_to_argument_properties(params)

        assert "$['user_id']" in result
        prop = result["$['user_id']"]
        assert isinstance(prop, AgentToolArgumentArgumentProperties)
        assert prop.is_sensitive is False
        assert prop.argument_path == "userId"

    def test_mixed_parameters(self):
        """Both static and argument parameters are converted correctly."""
        params = [
            AgentIntegrationToolParameter(
                name="base_url",
                type="string",
                value="https://api.example.com",
                field_location="body",
                field_variant="static",
            ),
            AgentIntegrationToolParameter(
                name="token",
                type="string",
                value="{{authToken}}",
                field_location="header",
                field_variant="argument",
            ),
        ]

        result = convert_integration_parameters_to_argument_properties(params)

        assert len(result) == 2

        assert "$['base_url']" in result
        static_prop = result["$['base_url']"]
        assert isinstance(static_prop, AgentToolStaticArgumentProperties)
        assert static_prop.value == "https://api.example.com"

        assert "$['token']" in result
        arg_prop = result["$['token']"]
        assert isinstance(arg_prop, AgentToolArgumentArgumentProperties)
        assert arg_prop.argument_path == "authToken"

    def test_parameter_without_field_variant_skipped(self):
        """Parameters with no fieldVariant are skipped."""
        params = [
            AgentIntegrationToolParameter(
                name="search_query",
                type="string",
                value="test",
                field_location="query",
                # field_variant is None by default
            ),
            AgentIntegrationToolParameter(
                name="api_key",
                type="string",
                value="key-123",
                field_location="header",
                field_variant="static",
            ),
        ]

        result = convert_integration_parameters_to_argument_properties(params)

        assert len(result) == 1
        assert "$['search_query']" not in result
        assert "$['api_key']" in result

    def test_empty_parameters(self):
        """Empty list returns empty dict."""
        result = convert_integration_parameters_to_argument_properties([])

        assert result == {}
        assert isinstance(result, dict)

    def test_nested_static_parameter_has_bracket_notation_key(self):
        """Nested static param produces bracket-notation JSONPath key."""
        params = [
            AgentIntegrationToolParameter(
                name="attachment.title",
                type="string",
                value="Custom title",
                field_location="body",
                field_variant="static",
            ),
        ]

        result = convert_integration_parameters_to_argument_properties(params)

        assert "$['attachment']['title']" in result
        prop = result["$['attachment']['title']"]
        assert isinstance(prop, AgentToolStaticArgumentProperties)
        assert prop.value == "Custom title"

    def test_nested_argument_parameter_has_bracket_notation_key(self):
        """Nested argument param produces bracket-notation JSONPath key."""
        params = [
            AgentIntegrationToolParameter(
                name="attachment.title_link",
                type="string",
                value="{{opportunityID}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        result = convert_integration_parameters_to_argument_properties(params)

        assert "$['attachment']['title_link']" in result
        prop = result["$['attachment']['title_link']"]
        assert isinstance(prop, AgentToolArgumentArgumentProperties)
        assert prop.argument_path == "opportunityID"

    def test_array_notation_parameter_has_bracket_notation_key(self):
        """Array-notation param produces bracket-notation JSONPath key with wildcard."""
        params = [
            AgentIntegrationToolParameter(
                name="attachments[*].text",
                type="string",
                value="fixed text",
                field_location="body",
                field_variant="static",
            ),
        ]

        result = convert_integration_parameters_to_argument_properties(params)

        assert "$['attachments'][*]['text']" in result
        prop = result["$['attachments'][*]['text']"]
        assert isinstance(prop, AgentToolStaticArgumentProperties)
        assert prop.is_sensitive is False
        assert prop.value == "fixed text"

    def test_argument_parameter_with_invalid_template_raises(self):
        """Malformed template raises AgentStartupError."""
        params = [
            AgentIntegrationToolParameter(
                name="bad_param",
                type="string",
                value="not_a_template",
                field_location="body",
                field_variant="argument",
            ),
        ]

        with pytest.raises(AgentStartupError):
            convert_integration_parameters_to_argument_properties(params)


class TestStripTemplateEnumsFromSchema:
    """Test cases for strip_template_enums_from_schema function."""

    def test_strips_enum_with_single_template_value(self):
        """An enum containing only a single template value is removed entirely."""
        schema = {
            "type": "object",
            "properties": {
                "opportunityID": {
                    "type": "string",
                    "enum": ["{{opportunityID}}"],
                }
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="opportunityID",
                type="string",
                value="{{opportunityID}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert "enum" not in result["properties"]["opportunityID"]

    def test_preserves_enum_without_templates(self):
        """An enum with only real values is left unchanged."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "closed"],
                }
            },
        }

        result = strip_template_enums_from_schema(schema, [])

        assert result["properties"]["status"]["enum"] == ["open", "closed"]

    def test_strips_template_preserves_real_values_in_mixed_enum(self):
        """Templates are removed but real values are kept in a mixed enum."""
        schema = {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["real", "{{template}}"],
                }
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="kind",
                type="string",
                value="{{template}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert result["properties"]["kind"]["enum"] == ["real"]

    def test_strips_empty_enum_after_removal(self):
        """When all enum values are templates the enum key is removed."""
        schema = {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["{{a}}", "{{b}}"],
                }
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="field",
                type="string",
                value="{{a}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert "enum" not in result["properties"]["field"]

    def test_handles_schema_without_properties(self):
        """A schema with no properties key is returned unchanged."""
        schema = {"type": "object"}

        result = strip_template_enums_from_schema(schema, [])

        assert result == {"type": "object"}

    def test_does_not_mutate_original_schema(self):
        """The original schema object must not be modified."""
        schema = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "enum": ["{{id}}"],
                }
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="id",
                type="string",
                value="{{id}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        id_field = schema["properties"]["id"]  # type: ignore[index]
        original_enum = id_field["enum"][:]
        strip_template_enums_from_schema(schema, parameters)

        assert id_field["enum"] == original_enum

    def test_handles_non_string_enum_values(self):
        """Non-string enum values are never treated as templates."""
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                }
            },
        }

        result = strip_template_enums_from_schema(schema, [])

        assert result["properties"]["count"]["enum"] == [1, 2, 3]

    def test_strips_enum_only_for_argument_variant_top_level(self):
        """Only argument-variant param's enum is stripped; dynamic param's enum is preserved."""
        schema = {
            "type": "object",
            "properties": {
                "opportunityID": {
                    "type": "string",
                    "enum": ["{{opportunityID}}"],
                },
                "parse": {
                    "type": "string",
                    "enum": ["none", "full"],
                },
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="opportunityID",
                type="string",
                value="{{opportunityID}}",
                field_location="body",
                field_variant="argument",
            ),
            AgentIntegrationToolParameter(
                name="parse",
                type="string",
                value="{{prompt}}",
                field_location="body",
                field_variant="dynamic",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert "enum" not in result["properties"]["opportunityID"]
        assert result["properties"]["parse"]["enum"] == ["none", "full"]

    def test_preserves_static_variant_enum(self):
        """Static-variant param's enum is never stripped even if values look like templates."""
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "enum": ["Custom title"],
                },
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="title",
                type="string",
                value="Custom title",
                field_location="body",
                field_variant="static",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert result["properties"]["title"]["enum"] == ["Custom title"]

    def test_strips_enum_on_nested_argument_field(self):
        """Argument-variant param's enum is stripped even when nested inside objects."""
        schema = {
            "type": "object",
            "properties": {
                "attachment": {
                    "type": "object",
                    "properties": {
                        "title_link": {
                            "type": "string",
                            "enum": ["{{opportunityID}}"],
                        },
                        "title": {
                            "type": "string",
                            "enum": ["Custom title"],
                        },
                    },
                },
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="attachment.title_link",
                type="string",
                value="{{opportunityID}}",
                field_location="body",
                field_variant="argument",
            ),
            AgentIntegrationToolParameter(
                name="attachment.title",
                type="string",
                value="Custom title",
                field_location="body",
                field_variant="static",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert (
            "enum" not in result["properties"]["attachment"]["properties"]["title_link"]
        )
        assert result["properties"]["attachment"]["properties"]["title"]["enum"] == [
            "Custom title"
        ]

    def test_strips_enum_on_array_nested_field(self):
        """Argument-variant param inside an array is correctly navigated and stripped."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["{{status}}", "active"],
                            },
                        },
                    },
                },
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="items[*].status",
                type="string",
                value="{{status}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert result["properties"]["items"]["items"]["properties"]["status"][
            "enum"
        ] == ["active"]

    def test_handles_ref_resolution(self):
        """Argument-variant param with $ref in schema path is resolved and enum stripped."""
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "$ref": "#/definitions/Config",
                },
            },
            "definitions": {
                "Config": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["{{mode}}", "auto"],
                        },
                    },
                },
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="config.mode",
                type="string",
                value="{{mode}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        # The $ref is inlined and modified on the inlined copy
        config_props = result["properties"]["config"]["properties"]
        assert config_props["mode"]["enum"] == ["auto"]

    def test_skips_argument_param_when_schema_path_not_found(self):
        """If the schema path for an argument param doesn't exist, skip silently."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        parameters = [
            AgentIntegrationToolParameter(
                name="nonexistent.field",
                type="string",
                value="{{val}}",
                field_location="body",
                field_variant="argument",
            ),
        ]

        result = strip_template_enums_from_schema(schema, parameters)

        assert result == schema


class TestCreateIntegrationToolWithArgumentProperties:
    """Test cases for create_integration_tool with argument_properties support."""

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
            input_schema=None,
            output_schema=None,
            properties=None,
            **properties_kwargs,
        ):
            if properties is None:
                properties = base_properties_factory(**properties_kwargs)

            return AgentIntegrationToolResourceConfig(
                name=name,
                description=description,
                properties=properties,
                input_schema=input_schema
                or {"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema=output_schema,
            )

        return _create_resource

    @patch("uipath_langchain.agent.tools.integration_tool.UiPath")
    def test_tool_has_argument_properties_for_static_param(
        self, mock_uipath_cls, resource_factory
    ):
        """Tool created with a static parameter has argument_properties entry."""
        mock_uipath_cls.return_value = MagicMock()

        params = [
            AgentIntegrationToolParameter(
                name="api_key",
                type="string",
                value="secret-key-123",
                field_location="header",
                field_variant="static",
            ),
        ]
        resource = resource_factory(parameters=params)

        tool = create_integration_tool(resource)

        assert isinstance(tool, StructuredToolWithArgumentProperties)
        assert "$['api_key']" in tool.argument_properties
        prop = tool.argument_properties["$['api_key']"]
        assert isinstance(prop, AgentToolStaticArgumentProperties)
        assert prop.value == "secret-key-123"

    @patch("uipath_langchain.agent.tools.integration_tool.UiPath")
    def test_tool_strips_template_enum_from_schema(
        self, mock_uipath_cls, resource_factory
    ):
        """Template enum values are stripped from the tool's args_schema for argument-variant params."""
        mock_uipath_cls.return_value = MagicMock()

        input_schema = {
            "type": "object",
            "properties": {
                "opportunityID": {
                    "type": "string",
                    "enum": ["{{opportunityID}}"],
                },
                "name": {
                    "type": "string",
                },
            },
        }
        params = [
            AgentIntegrationToolParameter(
                name="opportunityID",
                type="string",
                value="{{opportunityID}}",
                field_location="body",
                field_variant="argument",
            ),
        ]
        resource = resource_factory(input_schema=input_schema, parameters=params)

        tool = create_integration_tool(resource)

        assert isinstance(tool.args_schema, type)
        schema = tool.args_schema.model_json_schema()
        opp_field = schema["properties"]["opportunityID"]
        assert "enum" not in opp_field
        for def_value in schema.get("$defs", {}).values():
            assert "{{opportunityID}}" not in def_value.get("enum", [])

    @patch("uipath_langchain.agent.tools.integration_tool.UiPath")
    def test_tool_with_no_static_params_has_empty_argument_properties(
        self, mock_uipath_cls, resource_factory
    ):
        """Tool with parameters that have no fieldVariant gets empty argument_properties."""
        mock_uipath_cls.return_value = MagicMock()

        params = [
            AgentIntegrationToolParameter(
                name="search_query",
                type="string",
                field_location="query",
                # field_variant is None by default
            ),
        ]
        resource = resource_factory(parameters=params)

        tool = create_integration_tool(resource)

        assert isinstance(tool, StructuredToolWithArgumentProperties)
        assert tool.argument_properties == {}


class TestParseIsParamName:
    """Test cases for _parse_is_param_name helper."""

    def test_simple_field(self):
        assert _param_name_to_segments("channel") == ["channel"]

    def test_nested_field(self):
        assert _param_name_to_segments("attachment.title") == ["attachment", "title"]

    def test_deeply_nested_field(self):
        assert _param_name_to_segments("metadata.event_payload.id") == [
            "metadata",
            "event_payload",
            "id",
        ]

    def test_array_notation(self):
        assert _param_name_to_segments("attachments[*]") == ["attachments", "*"]

    def test_array_with_nested_field(self):
        assert _param_name_to_segments("attachments[*].text") == [
            "attachments",
            "*",
            "text",
        ]

    def test_deeply_nested_with_multiple_arrays(self):
        assert _param_name_to_segments("attachments[*].actions[*].confirm.text") == [
            "attachments",
            "*",
            "actions",
            "*",
            "confirm",
            "text",
        ]


class TestIsParamNameToJsonpath:
    """Test cases for _is_param_name_to_jsonpath helper."""

    def test_simple_field(self):
        assert _is_param_name_to_jsonpath("channel") == "$['channel']"

    def test_nested_field(self):
        assert (
            _is_param_name_to_jsonpath("attachment.title") == "$['attachment']['title']"
        )

    def test_deeply_nested_field(self):
        assert _is_param_name_to_jsonpath("metadata.event_payload.id") == (
            "$['metadata']['event_payload']['id']"
        )

    def test_array_notation(self):
        assert _is_param_name_to_jsonpath("attachments[*]") == "$['attachments'][*]"

    def test_array_with_nested_field(self):
        assert _is_param_name_to_jsonpath("attachments[*].text") == (
            "$['attachments'][*]['text']"
        )

    def test_deeply_nested_with_multiple_arrays(self):
        assert (
            _is_param_name_to_jsonpath("attachments[*].actions[*].confirm.text")
            == "$['attachments'][*]['actions'][*]['confirm']['text']"
        )

    def test_escapes_single_quotes(self):
        assert _is_param_name_to_jsonpath("it's_field") == "$['it\\'s_field']"

    def test_escapes_backslashes(self):
        assert _is_param_name_to_jsonpath("path\\to") == "$['path\\\\to']"
