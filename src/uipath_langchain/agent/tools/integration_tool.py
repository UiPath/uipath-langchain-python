"""Process tool creation for UiPath process execution."""

from __future__ import annotations

from typing import Any, Dict

from jsonschema_pydantic_converter import transform as create_model
from langchain_core.tools import StructuredTool
from pydantic import TypeAdapter
from uipath.agent.models.agent import AgentIntegrationToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.connections import ActivityMetadata, ActivityParameterLocationInfo

from .utils import sanitize_tool_name


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

    input_model: Any = create_model(resource.input_schema)
    # note: IS tools output schemas were recently added and are most likely not present in all resources
    output_model: Any = (
        create_model(resource.output_schema)
        if resource.output_schema
        else create_model({"type": "object", "properties": {}})
    )

    sdk = UiPath()

    def sanitize_for_serialization(args: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Pydantic models in args to dicts."""
        converted_args: Dict[str, Any] = {}
        for key, value in args.items():
            # handle Pydantic model
            if hasattr(value, "model_dump"):
                converted_args[key] = value.model_dump()

            elif isinstance(value, list):
                # handle list of Pydantic models
                converted_list = []
                for item in value:
                    if hasattr(item, "model_dump"):
                        converted_list.append(item.model_dump())
                    else:
                        converted_list.append(item)
                converted_args[key] = converted_list

            # handle regular value or unexpected type
            else:
                converted_args[key] = value
        return converted_args

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def integration_tool_fn(**kwargs: Any) -> output_model:
        try:
            result = await sdk.connections.invoke_activity_async(
                activity_metadata=activity_metadata,
                connection_id=connection_id,
                activity_input=sanitize_for_serialization(kwargs),
            )
        except Exception:
            raise

        return TypeAdapter(output_model).validate_python(result)

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=integration_tool_fn,
    )

    tool.__dict__["OutputType"] = output_model

    return tool
