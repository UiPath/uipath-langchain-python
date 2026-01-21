"""Handles static arguments for tool calls."""

import copy
import logging
from typing import Any, Iterator, Mapping

from jsonpath_ng import parse  # type: ignore[import-untyped]
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentIntegrationToolParameter,
    AgentIntegrationToolResourceConfig,
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolStaticArgumentProperties,
    AgentToolTextBuilderArgumentProperties,
    BaseAgentResourceConfig,
)
from uipath.agent.utils.text_tokens import build_string_from_tokens

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.schema_editing import (
    SchemaModificationError,
    apply_static_value_to_schema,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from .utils import sanitize_dict_for_serialization

logger = logging.getLogger(__name__)


class ToolStaticArgument(BaseModel):
    """Tool static argument model."""

    value: Any
    is_sensitive: bool


def _resolve_argument_properties_to_static_arguments(
    argument_properties: Mapping[str, AgentToolArgumentProperties],
    agent_input: dict[str, Any],
) -> dict[str, ToolStaticArgument]:
    """Resolves the different variants of argument properties to static arguments."""

    def resolve_to_static(
        props: AgentToolArgumentProperties,
    ) -> ToolStaticArgument | None:
        """Resolves argument and textBuilder variants to static."""
        match props:
            case AgentToolStaticArgumentProperties():
                return ToolStaticArgument(
                    value=props.value, is_sensitive=props.is_sensitive
                )
            case AgentToolArgumentArgumentProperties():
                agent_argument = parse(props.argument_path).find(agent_input)
                if not agent_argument:
                    return None
                else:
                    argument_value = agent_argument[0].value
                return ToolStaticArgument(
                    value=argument_value, is_sensitive=props.is_sensitive
                )
            case AgentToolTextBuilderArgumentProperties():
                text_value = build_string_from_tokens(props.tokens, agent_input)
                return ToolStaticArgument(
                    value=text_value, is_sensitive=props.is_sensitive
                )
            case _:
                raise ValueError(f"Unsupported argument property type: {type(props)}")

    def deduplicate_argument_properties(
        properties: Mapping[str, AgentToolArgumentProperties],
    ) -> Iterator[tuple[str, AgentToolArgumentProperties]]:
        """Skips more specific argument properties. In effect, prioritizes parent paths over child paths."""

        sorted_paths = sorted(properties.keys())
        for i, json_path in enumerate(sorted_paths):
            if i > 0 and json_path.startswith(sorted_paths[i - 1]):
                continue
            yield json_path, properties[json_path]

    static_args: dict[str, ToolStaticArgument] = {}
    for json_path, props in deduplicate_argument_properties(argument_properties):
        static_arg = resolve_to_static(props)
        if static_arg is not None:
            static_args[json_path] = static_arg
    return static_args


def apply_static_argument_properties_to_schema(
    tool: StructuredToolWithArgumentProperties,
    agent_input: dict[str, Any],
) -> StructuredToolWithArgumentProperties:
    """Modify tool schema based on static argumentProperties.

    Args:
        tool: The tool to modify
        agent_input: The agent input to use for resolving argument variants
    """
    if not tool.argument_properties:
        return tool

    if isinstance(tool.args_schema, dict):
        modified_json_schema = copy.deepcopy(tool.args_schema)
    elif tool.args_schema and issubclass(tool.args_schema, BaseModel):
        modified_json_schema = tool.args_schema.model_json_schema()
    else:
        return tool

    static_args = _resolve_argument_properties_to_static_arguments(
        tool.argument_properties, agent_input
    )
    for json_path, static_arg in static_args.items():
        try:
            apply_static_value_to_schema(
                modified_json_schema,
                json_path,
                static_arg.value,
                static_arg.is_sensitive,
            )
        except SchemaModificationError as e:
            logger.warning(
                f"Skipping invalid static argument path '{json_path}' for tool '{tool.name}': {e}"
            )

    modified_tool = tool.model_copy(deep=True)
    modified_tool.args_schema = create_model(modified_json_schema)

    return modified_tool


def resolve_static_args(
    resource: BaseAgentResourceConfig,
    agent_input: dict[str, Any],
) -> dict[str, Any]:
    """Resolves static arguments for a given resource with a given input.

    Args:
        resource: The agent resource configuration.
        input: The input arguments passed to the agent.

    Returns:
        A dictionary of expanded arguments to be used in the tool call.
    """

    if isinstance(resource, AgentIntegrationToolResourceConfig):
        return resolve_integration_static_args(
            resource.properties.parameters, agent_input
        )
    elif hasattr(resource, "argument_properties"):
        # TODO: MCP tools don't inherit from BaseAgentResourceConfig; will need to handle separately
        static_arguments = _resolve_argument_properties_to_static_arguments(
            resource.argument_properties, agent_input
        )
        return {
            json_path: static_argument.value
            for json_path, static_argument in static_arguments.items()
        }
    else:
        return {}  # to be implemented for other resource types in the future


def resolve_integration_static_args(
    parameters: list[AgentIntegrationToolParameter],
    agent_input: dict[str, Any],
) -> dict[str, Any]:
    """Resolves static arguments for an integration tool resource.

    Args:
        resource: The AgentIntegrationToolResourceConfig instance.
        input: The input arguments passed to the agent.

    Returns:
        A dictionary of expanded static arguments for the integration tool.
    """

    static_args: dict[str, Any] = {}
    for param in parameters:
        value = None

        # static parameter, use the defined static value
        if param.field_variant == "static":
            value = param.value
        # argument parameter, extract value from agent input
        elif param.field_variant == "argument":
            if (
                not isinstance(param.value, str)
                or not param.value.startswith("{{")
                or not param.value.endswith("}}")
            ):
                raise ValueError(
                    f"Parameter value must be in the format '{{argument_name}}' when field_variant is 'argument', got {param.value}"
                )
            arg_name = param.value[2:-2].strip()
            # currently only support top-level arguments
            value = agent_input.get(arg_name)

        if value is not None:
            static_args[param.name] = value

    return static_args


def apply_static_args(
    static_args: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Applies static arguments to the given input arguments.

    Args:
        static_args: Dictionary of static arguments {json_path: value} to apply.
        kwargs: Original input arguments to the tool.

    Returns:
        Merged input arguments with static arguments applied.
    """

    sanitized_args = sanitize_dict_for_serialization(kwargs)
    for json_path, value in static_args.items():
        expr = parse(json_path)

        if json_path.endswith("[*]"):
            # This targets the array itself
            array_json_path = json_path[:-3]
            array_expr = parse(array_json_path)
            actual_array = array_expr.find(sanitized_args)
            actual_value = actual_array[0].value
            if isinstance(actual_value, list) and len(actual_value) == 0:
                # The array is empty. Updating it with jsonpath will leave it empty.
                # We instead replace the empty array with a single static value
                array_expr.update_or_create(sanitized_args, [value])
                return sanitized_args

        expr.update_or_create(sanitized_args, value)

    return sanitized_args


def handle_static_args(
    resource: BaseAgentResourceConfig, state: BaseModel, input_args: dict[str, Any]
) -> dict[str, Any]:
    """Resolves and applies static arguments for a tool call.
    Args:
        resource: The agent resource configuration.
        runtime: The tool runtime providing the current state.
        input_args: The original input arguments to the tool.
    Returns:
        A dictionary of input arguments with static arguments applied.
    """

    static_args = resolve_static_args(resource, dict(state))
    sanitized_static_args = sanitize_dict_for_serialization(static_args)
    merged_args = apply_static_args(sanitized_static_args, input_args)
    return merged_args
