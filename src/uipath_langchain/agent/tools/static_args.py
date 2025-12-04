"""Handles static arguments for tool calls."""

from typing import Any, Dict, List

from jsonpath_ng import parse  # type: ignore[import-untyped]
from langchain.tools import ToolRuntime
from uipath.agent.models.agent import (
    AgentIntegrationToolParameter,
    AgentIntegrationToolResourceConfig,
    BaseAgentResourceConfig,
)


def resolve_static_args(
    resource: BaseAgentResourceConfig,
    agent_input: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolves static arguments for a given resource with a given input.

    Args:
        resource: The agent resource configuration.
        input: Othe input arguments passed to the agent.

    Returns:
        A dictionary of expanded arguments to be used in the tool call.
    static_args: Dict[str, Any] = {}
    """

    if isinstance(resource, AgentIntegrationToolResourceConfig):
        return resolve_integration_static_args(
            resource.properties.parameters, agent_input
        )
    else:
        return {}  # to be implemented for other resource types in the future


def resolve_integration_static_args(
    parameters: List[AgentIntegrationToolParameter],
    agent_input: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolves static arguments for an integration tool resource.

    Args:
        resource: The AgentIntegrationToolResourceConfig instance.
        input: The input arguments passed to the agent.

    Returns:
        A dictionary of expanded static arguments for the integration tool.
    """

    static_args: Dict[str, Any] = {}
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


def apply_static_args(
    static_args: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Applies static arguments to the given input arguments.

    Args:
        static_args: Dictionary of static arguments {json_path: value} to apply.
        kwargs: Original input arguments to the tool.

    Returns:
        Merged input arguments with static arguments applied.
    """

    sanitized_args = sanitize_for_serialization(kwargs)
    for json_path, value in static_args.items():
        expr = parse(json_path)
        expr.update_or_create(sanitized_args, value)

    return sanitized_args


def handle_static_args(
    resource: BaseAgentResourceConfig, runtime: ToolRuntime, input_args: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolves and applies static arguments for a tool call.
    Args:
        resource: The agent resource configuration.
        runtime: The tool runtime providing the current state.
        input_args: The original input arguments to the tool.
    Returns:
        A dictionary of input arguments with static arguments applied.
    """

    static_args = resolve_static_args(resource, dict(runtime.state))
    merged_args = apply_static_args(static_args, input_args)
    return merged_args
