"""Process tool creation for UiPath process execution."""

import copy
from typing import Any

from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentIntegrationToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.connections import ActivityMetadata, ActivityParameterLocationInfo

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin
from uipath_langchain.agent.wrappers.static_args_wrapper import get_static_args_wrapper

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_dict_for_serialization, sanitize_tool_name


class StructuredToolWithStaticArgs(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def remove_asterisk_from_properties(fields: dict[str, Any]) -> dict[str, Any]:
    """
    Fix bug in integration service.
    """
    fields = copy.deepcopy(fields)

    def fix_types(props: dict[str, Any]) -> None:
        type_ = props.get("type", None)
        if "$ref" in props:
            props["$ref"] = props["$ref"].replace("[*]", "")
        if type_ == "object":
            properties = {}
            for k, v in props.get("properties", {}).items():
                # Remove asterisks!
                k = k.replace("[*]", "")
                properties[k] = v
                if isinstance(v, dict):
                    fix_types(v)
            if "properties" in props:
                props["properties"] = properties
        if type_ == "array":
            fix_types(props.get("items", {}))

    definitions = {}
    for k, value in fields.get("$defs", fields.get("definitions", {})).items():
        k = k.replace("[*]", "")
        definitions[k] = value
        fix_types(value)
    if "definitions" in fields:
        fields["definitions"] = definitions

    fix_types(fields)
    return fields


def extract_top_level_field(param_name: str) -> str:
    """Extract the top-level field name from a jsonpath parameter name.

    Examples:
        metadata.field.test -> metadata
        attachments[*] -> attachments
        attachments[0].filename -> attachments
        simple_field -> simple_field
    """
    # Split by '.' to get the first part
    first_part = param_name.split(".")[0]

    # Remove array notation if present (e.g., "attachments[*]" -> "attachments")
    if "[" in first_part:
        first_part = first_part.split("[")[0]

    return first_part


def convert_to_activity_metadata(
    resource: AgentIntegrationToolResourceConfig,
) -> ActivityMetadata:
    """Convert AgentIntegrationToolResourceConfig to ActivityMetadata."""

    # normalize HTTP method (GETBYID -> GET)
    http_method = resource.properties.method
    if http_method == "GETBYID":
        http_method = "GET"

    param_location_info = ActivityParameterLocationInfo()
    # because of nested fields and array notation, use a set to avoid duplicates
    body_fields_set = set()

    # mapping parameter locations
    for param in resource.properties.parameters:
        param_name = param.name
        field_location = param.field_location

        if field_location == "query":
            param_location_info.query_params.append(param_name)
        elif field_location == "path":
            param_location_info.path_params.append(param_name)
        elif field_location == "header":
            param_location_info.header_params.append(param_name)
        elif field_location in ("multipart", "file"):
            param_location_info.multipart_params.append(param_name)
        elif field_location == "body":
            # extract top-level field from jsonpath parameter name
            top_level_field = extract_top_level_field(param_name)
            body_fields_set.add(top_level_field)
        else:
            # default to body field - extract top-level field
            top_level_field = extract_top_level_field(param_name)
            body_fields_set.add(top_level_field)

    param_location_info.body_fields = list(body_fields_set)

    # determine content type
    content_type = "application/json"
    if resource.properties.body_structure is not None:
        shorthand_type = resource.properties.body_structure.get("contentType", "json")
        if shorthand_type == "multipart":
            content_type = "multipart/form-data"

    return ActivityMetadata(
        object_path=resource.properties.tool_path,
        method_name=http_method,
        content_type=content_type,
        parameter_location_info=param_location_info,
    )


def create_integration_tool(
    resource: AgentIntegrationToolResourceConfig,
) -> StructuredTool:
    """Creates a StructuredTool for invoking an Integration Service connector activity."""
    tool_name: str = sanitize_tool_name(resource.name)
    if resource.properties.connection.id is None:
        raise ValueError("Connection ID cannot be None for integration tool.")
    connection_id: str = resource.properties.connection.id

    activity_metadata = convert_to_activity_metadata(resource)

    input_model = create_model(resource.input_schema)
    # note: IS tools output schemas were recently added and are most likely not present in all resources
    output_model: Any = (
        create_model(remove_asterisk_from_properties(resource.output_schema))
        if resource.output_schema
        else create_model({"type": "object", "properties": {}})
    )

    sdk = UiPath()

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=resource.properties.example_calls,
    )
    async def integration_tool_fn(**kwargs: Any):
        try:
            result = await sdk.connections.invoke_activity_async(
                activity_metadata=activity_metadata,
                connection_id=connection_id,
                activity_input=sanitize_dict_for_serialization(kwargs),
            )
        except Exception:
            raise

        return result

    wrapper = get_static_args_wrapper(resource)

    tool = StructuredToolWithStaticArgs(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=integration_tool_fn,
        output_type=output_model,
    )
    tool.set_tool_wrappers(awrapper=wrapper)

    return tool
