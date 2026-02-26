"""Process tool creation for UiPath process execution."""

import copy
import re
from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentIntegrationToolParameter,
    AgentIntegrationToolResourceConfig,
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolStaticArgumentProperties,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.connections import ActivityMetadata, ActivityParameterLocationInfo
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperReturnType,
)

from .structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from .utils import sanitize_dict_for_serialization, sanitize_tool_name


def convert_integration_parameters_to_argument_properties(
    parameters: list[AgentIntegrationToolParameter],
) -> dict[str, AgentToolArgumentProperties]:
    """Convert integration tool parameters to argument_properties format.

    Converts parameters with fieldVariant 'static' or 'argument' to the
    corresponding AgentToolArgumentProperties type. Parameters without
    a recognized fieldVariant are skipped.

    Args:
        parameters: List of integration tool parameters to convert.

    Returns:
        Dictionary mapping JSONPath keys to argument properties.

    Raises:
        AgentStartupError: If an argument variant parameter has a malformed template.
    """
    result: dict[str, AgentToolArgumentProperties] = {}

    for param in parameters:
        if param.field_variant == "static":
            key = _is_param_name_to_jsonpath(param.name)
            result[key] = AgentToolStaticArgumentProperties(
                is_sensitive=False,
                value=param.value,
            )
        elif param.field_variant == "argument":
            value_str = str(param.value) if param.value is not None else ""
            match = re.fullmatch(r"\{\{(.+?)\}\}", value_str)
            if not match:
                raise AgentStartupError(
                    code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
                    title="Malformed integration tool argument",
                    detail=f"Argument parameter '{param.name}' has malformed template: "
                    f"'{param.value}'. Expected format: '{{{{argName}}}}'",
                    category=UiPathErrorCategory.USER,
                )
            arg_name = match.group(1)
            key = _is_param_name_to_jsonpath(param.name)
            result[key] = AgentToolArgumentArgumentProperties(
                is_sensitive=False,
                argument_path=arg_name,
            )

    return result


_TEMPLATE_PATTERN = re.compile(r"^\{\{.*\}\}$")


def _param_name_to_segments(param_name: str) -> list[str]:
    """Parse an Integration Service dot-notation parameter name into path segments.

    Splits by '.' and expands '[*]' suffixes into a separate '*' wildcard segment.

    Examples:
        "channel"                             -> ["channel"]
        "attachment.title"                    -> ["attachment", "title"]
        "attachments[*].actions[*].confirm.text"
            -> ["attachments", "*", "actions", "*", "confirm", "text"]
    """
    segments: list[str] = []
    for part in param_name.split("."):
        if part.endswith("[*]"):
            segments.append(part[:-3])
            segments.append("*")
        else:
            segments.append(part)
    return segments


def _escape_jsonpath_property(name: str) -> str:
    """Escape a property name for bracket-notation JSONPath.

    Per the JSONPath bracket-notation spec, escapes are applied in order:
    1. Backslashes first: \\\\ -> \\\\\\\\
    2. Single quotes second: ' -> \\\\'
    """
    return name.replace("\\", "\\\\").replace("'", "\\'")


def _is_param_name_to_jsonpath(param_name: str) -> str:
    """Convert an IS dot-notation parameter name to bracket-notation JSONPath.

    Examples:
        "channel"            -> "$['channel']"
        "attachment.title"   -> "$['attachment']['title']"
        "attachments[*].text" -> "$['attachments'][*]['text']"
    """
    segments = _param_name_to_segments(param_name)
    parts: list[str] = []
    for seg in segments:
        if seg == "*":
            parts.append("[*]")
        else:
            parts.append(f"['{_escape_jsonpath_property(seg)}']")
    return "$" + "".join(parts)


def _resolve_schema_ref(
    root_schema: dict[str, Any], node: dict[str, Any]
) -> dict[str, Any]:
    """Resolve a $ref pointer in a JSON schema node.

    Returns the referenced definition if $ref is present,
    otherwise returns the node unchanged.
    """
    ref = node.get("$ref")
    if ref is None:
        return node
    parts = ref.lstrip("#/").split("/")
    current = root_schema
    for part in parts:
        current = current[part]
    return current


def _navigate_schema_to_field(
    root_schema: dict[str, Any], segments: list[str]
) -> dict[str, Any] | None:
    """Navigate a JSON schema to a leaf field using parsed path segments.

    Args:
        root_schema: The root schema (needed for $ref resolution).
        segments: Path segments from _parse_is_param_name.

    Returns:
        The schema dict of the leaf field, or None if the path doesn't exist.
    """
    current = root_schema
    for seg in segments:
        current = _resolve_schema_ref(root_schema, current)
        if seg == "*":
            items = current.get("items")
            if items is None:
                return None
            current = items
        else:
            props = current.get("properties")
            if props is None or seg not in props:
                return None
            current = props[seg]
    return _resolve_schema_ref(root_schema, current)


def strip_template_enums_from_schema(
    schema: dict[str, Any],
    parameters: list[AgentIntegrationToolParameter],
) -> dict[str, Any]:
    """Remove {{template}} enum values only from argument-variant parameter fields.

    For each parameter with fieldVariant 'argument', navigates the schema to the
    corresponding field (supporting nested objects, arrays, and $ref resolution)
    and strips enum values matching the {{...}} pattern.

    The function deep-copies the schema so the original is never mutated.

    Args:
        schema: A JSON-schema-style dictionary (the tool's inputSchema).
        parameters: List of integration tool parameters from resource.properties.

    Returns:
        A cleaned copy of the schema with template enum values removed
        only from argument-variant fields.
    """
    schema = copy.deepcopy(schema)

    for param in parameters:
        if param.field_variant != "argument":
            continue

        segments = _param_name_to_segments(param.name)
        field_schema = _navigate_schema_to_field(schema, segments)
        if field_schema is None:
            continue

        enum = field_schema.get("enum")
        if enum is None:
            continue

        cleaned = [
            v for v in enum if not (isinstance(v, str) and _TEMPLATE_PATTERN.match(v))
        ]
        if not cleaned:
            del field_schema["enum"]
        else:
            field_schema["enum"] = cleaned

    return schema


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

    # determine content type and json body section
    content_type = "application/json"
    json_body_section = None
    if resource.properties.body_structure is not None:
        shorthand_type = resource.properties.body_structure.get("contentType", "json")
        if shorthand_type == "multipart":
            content_type = "multipart/form-data"
        json_body_section = resource.properties.body_structure.get("jsonBodySection")

    return ActivityMetadata(
        object_path=resource.properties.tool_path,
        method_name=http_method,
        content_type=content_type,
        parameter_location_info=param_location_info,
        json_body_section=json_body_section,
    )


def create_integration_tool(
    resource: AgentIntegrationToolResourceConfig,
) -> StructuredTool:
    """Creates a StructuredTool for invoking an Integration Service connector activity."""
    tool_name: str = sanitize_tool_name(resource.name)
    if resource.properties.connection.id is None:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing connection ID",
            detail="Connection ID cannot be None for integration tool.",
            category=UiPathErrorCategory.SYSTEM,
        )
    connection_id: str = resource.properties.connection.id

    activity_metadata = convert_to_activity_metadata(resource)

    cleaned_input_schema = strip_template_enums_from_schema(
        resource.input_schema, resource.properties.parameters
    )
    input_model = create_model(cleaned_input_schema)
    # note: IS tools output schemas were recently added and are most likely not present in all resources
    output_model: Any = (
        create_model(remove_asterisk_from_properties(resource.output_schema))
        if resource.output_schema
        else create_model({"type": "object", "properties": {}})
    )

    argument_properties = convert_integration_parameters_to_argument_properties(
        resource.properties.parameters
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

    async def integration_tool_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        call["args"] = handle_static_args(resource, state, call["args"])
        return await tool.ainvoke(call)

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=integration_tool_fn,
        output_type=output_model,
        metadata={
            "tool_type": "integration",
            "display_name": resource.name,
            "connector_key": resource.properties.connection.id,
            "connector_name": resource.properties.connection.name,
        },
        argument_properties=argument_properties,
    )
    tool.set_tool_wrappers(awrapper=integration_tool_wrapper)

    return tool
