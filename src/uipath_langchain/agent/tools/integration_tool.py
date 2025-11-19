"""Process tool creation for UiPath process execution."""

from __future__ import annotations

from typing import Any, Type

from jsonschema_pydantic import jsonschema_to_pydantic  # type: ignore[import-untyped]
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath import UiPath
from uipath.agent.models.agent import AgentIntegrationToolResourceConfig
from uipath.models import ActivityMetadata, ActivityParameterLocationInfo

from .utils import sanitize_tool_name


def convert_to_integration_service_metadata(
    resource: AgentIntegrationToolResourceConfig,
) -> ActivityMetadata:
    """Convert AgentIntegrationToolResourceConfig to ActivityMetadata."""

    # normalize HTTP method (GETBYID -> GET)
    http_method = resource.properties.method
    if http_method == "GETBYID":
        http_method = "GET"

    # mapping parameter locations
    param_location_info = ActivityParameterLocationInfo()
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
            param_location_info.body_fields.append(param_name)
        else:
            # default to body field
            param_location_info.body_fields.append(param_name)

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

    activity_metadata = convert_to_integration_service_metadata(resource)

    input_model: Type[BaseModel] = jsonschema_to_pydantic(resource.input_schema)
    # note: IS tools output schemas were recently added and are most likely not present in all resources
    output_model: Type[BaseModel] | None = (
        jsonschema_to_pydantic(resource.output_schema)
        if resource.output_schema
        else None
    )

    sdk = UiPath()

    async def integration_tool_fn(**kwargs: Any):
        try:
            result = await sdk.connections.invoke_activity_async(
                activity_metadata=activity_metadata,
                connection_id=connection_id,
                activity_input=kwargs,
            )
        except Exception:
            raise

        return result

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=integration_tool_fn,
    )

    tool.__dict__["OutputType"] = output_model

    return tool
