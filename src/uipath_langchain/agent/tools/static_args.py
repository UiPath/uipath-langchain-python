"""Handles static arguments for tool calls."""

import copy
import logging
import re
from typing import Any, Iterator, Mapping, Sequence, TypeVar

from jsonpath_ng import parse  # type: ignore[import-untyped]
from jsonpath_ng.exceptions import JsonPathParserError  # type: ignore[import-untyped]
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from typing_extensions import deprecated
from uipath.agent.models.agent import (
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolArrayBuilderArgumentProperties,
    AgentToolStaticArgumentProperties,
    AgentToolTextBuilderArgumentProperties,
)
from uipath.agent.utils.text_tokens import build_string_from_tokens
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.utils import extract_input_data_from_state
from uipath_langchain.agent.tools.schema_editing import (
    InvalidStaticArgError,
    SchemaNavigationError,
    apply_static_value_to_schema,
)

from .utils import sanitize_dict_for_serialization

logger = logging.getLogger(__name__)


class ArgumentPropertiesMixin:
    argument_properties: dict[str, AgentToolArgumentProperties]


class ToolStaticArgument(BaseModel):
    """Tool static argument model."""

    value: Any
    display_value: Any
    is_sensitive: bool


_INDEX_AND_REST_REGEX = re.compile(r"^\[(\d+)\](.*)$")

_SENSITIVE_ITEM_PLACEHOLDER = "<hidden>"


def _resolve_argument_properties(
    argument_properties: Mapping[str, AgentToolArgumentProperties],
    agent_input: dict[str, Any],
    tool_name: str | None = None,
) -> dict[str, ToolStaticArgument]:
    """Resolves the different variants of argument properties to static arguments."""

    def resolve_to_static(
        props: AgentToolArgumentProperties,
        json_path: str,
    ) -> ToolStaticArgument | None:
        """Resolves argument, textBuilder, and arrayBuilder variants to static."""
        match props:
            case AgentToolStaticArgumentProperties():
                return ToolStaticArgument(
                    value=props.value,
                    display_value=props.value,
                    is_sensitive=props.is_sensitive,
                )
            case AgentToolArgumentArgumentProperties():
                try:
                    argument_expr = parse(props.argument_path)
                except JsonPathParserError as e:
                    tool_ref = f" of tool '{tool_name}'" if tool_name else ""
                    raise AgentRuntimeError(
                        code=AgentRuntimeErrorCode.INVALID_STATIC_ARGUMENT,
                        title="Malformed argument path in tool configuration",
                        detail=(
                            f"The argument binding '{json_path}'{tool_ref} has a "
                            f"malformed JSONPath expression {props.argument_path!r}: {e} "
                            "Fix the argument path in the agent configuration."
                        ),
                        category=UiPathErrorCategory.SYSTEM,
                    ) from e
                agent_argument = argument_expr.find(agent_input)
                if not agent_argument:
                    return None
                else:
                    argument_value = agent_argument[0].value
                return ToolStaticArgument(
                    value=argument_value,
                    display_value=argument_value,
                    is_sensitive=props.is_sensitive,
                )
            case AgentToolTextBuilderArgumentProperties():
                text_value = build_string_from_tokens(props.tokens, agent_input)
                return ToolStaticArgument(
                    value=text_value,
                    display_value=text_value,
                    is_sensitive=props.is_sensitive,
                )
            case AgentToolArrayBuilderArgumentProperties():
                return resolve_arraybuilder(json_path, argument_properties)
            case _:
                raise ValueError(f"Unsupported argument property type: {type(props)}")

    def resolve_arraybuilder(
        base_path: str,
        argument_properties: Mapping[str, AgentToolArgumentProperties],
    ) -> ToolStaticArgument:
        """Build an array value from arrayBuilder indexed children.

        Only direct indexed children ``base_path[N]`` are considered; entries
        with other nested properties are out of scope and silently skipped."""

        base_with_bracket = base_path + "["
        direct_children: dict[int, AgentToolArgumentProperties] = {}
        max_index = -1

        for path, props in argument_properties.items():
            if not path.startswith(base_with_bracket):
                continue
            match = _INDEX_AND_REST_REGEX.match(path[len(base_path) :])
            if not match or match.group(2) != "":
                continue
            idx = int(match.group(1))
            direct_children[idx] = props
            if idx > max_index:
                max_index = idx

        runtime_items: list[Any] = []
        display_items: list[Any] = []
        for i in range(max_index + 1):
            item_props = direct_children.get(i)
            if item_props is None:
                runtime_items.append(None)
                display_items.append(None)
                continue
            resolved = resolve_to_static(item_props, f"{base_path}[{i}]")
            if resolved is None:
                runtime_items.append(None)
                display_items.append(None)
            else:
                runtime_items.append(resolved.value)
                display_value = (
                    _SENSITIVE_ITEM_PLACEHOLDER
                    if resolved.is_sensitive
                    else resolved.display_value
                )
                display_items.append(display_value)

        return ToolStaticArgument(
            value=runtime_items, display_value=display_items, is_sensitive=False
        )

    def deduplicate_argument_properties(
        properties: Mapping[str, AgentToolArgumentProperties],
    ) -> Iterator[tuple[str, AgentToolArgumentProperties]]:
        """Skips more specific argument properties. In effect, prioritizes parent paths over child paths."""

        last_yielded: str | None = None
        for json_path in sorted(properties.keys()):
            if last_yielded is not None and json_path.startswith(last_yielded):
                continue
            yield json_path, properties[json_path]
            last_yielded = json_path

    static_args: dict[str, ToolStaticArgument] = {}
    for json_path, props in deduplicate_argument_properties(argument_properties):
        static_arg = resolve_to_static(props, json_path)
        if static_arg is not None:
            static_args[json_path] = static_arg
    return static_args


ToolT = TypeVar("ToolT", bound=StructuredTool)


def _apply_static_arguments_to_schema(
    tool: ToolT,
    static_args: dict[str, ToolStaticArgument],
) -> tuple[ToolT, set[str]]:
    """Modify tool schema based on pre-resolved static arguments.

    Args:
        tool: The tool to modify
        static_args: The mapping from JSON paths to static arguments

    Returns:
        The schema-modified tool and the set of json paths that were applied to
        the schema. Paths that cannot be applied are skipped.
    """
    if not static_args:
        return tool, set()

    if isinstance(tool.args_schema, dict):
        modified_json_schema = copy.deepcopy(tool.args_schema)
    elif tool.args_schema and issubclass(tool.args_schema, BaseModel):
        modified_json_schema = tool.args_schema.model_json_schema()
    else:
        return tool, set(static_args)

    applied_paths: set[str] = set()
    for json_path, static_arg in static_args.items():
        try:
            apply_static_value_to_schema(
                modified_json_schema,
                json_path,
                static_arg.display_value,
                static_arg.is_sensitive,
            )
            applied_paths.add(json_path)
        except InvalidStaticArgError as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.INVALID_STATIC_ARGUMENT,
                title="Invalid static argument value",
                detail=f"At path '{json_path}' for tool '{tool.name}': {e}",
                category=UiPathErrorCategory.USER,
            ) from e
        except SchemaNavigationError as e:
            logger.warning(
                f"Skipping invalid static argument path '{json_path}' for tool '{tool.name}': {e}"
            )

    modified_tool = tool.model_copy(deep=True)
    modified_tool.args_schema = create_model(modified_json_schema)

    return modified_tool, applied_paths


@deprecated(
    "Use StaticArgsHandler to modify LLM tool calls directly."
    "Applying static args in the tool node conflicts with guardrails"
)
def resolve_static_args(
    tool: ArgumentPropertiesMixin,
    agent_input: dict[str, Any],
) -> dict[str, Any]:
    """Resolves static arguments for a given resource with a given input.

    Args:
        tool: The tool with argument_properties.
        agent_input: The input arguments passed to the agent.

    Returns:
        A dictionary of expanded arguments to be used in the tool call.
    """

    static_arguments = _resolve_argument_properties(
        tool.argument_properties, agent_input
    )
    return {
        json_path: static_argument.value
        for json_path, static_argument in static_arguments.items()
    }


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
                continue

        expr.update_or_create(sanitized_args, value)

    return sanitized_args


class StaticArgsHandler:
    """Resolves and applies static args to tool schemas and tool calls."""

    _sanitized_static_values: dict[str, dict[str, Any]] | None
    _processed_tools: list[BaseTool] | None

    def __init__(self) -> None:
        self._sanitized_static_values = None
        self._processed_tools = None

    def initialize(
        self,
        tools: Sequence[BaseTool],
        state: BaseModel,
        input_schema: type[BaseModel],
    ) -> list[BaseTool]:
        """Resolves static args with the agent input and returns the schema-modified tools. Initializes once."""
        if self._processed_tools is not None:
            return self._processed_tools

        agent_input = extract_input_data_from_state(state, input_schema)

        self._processed_tools = []
        self._sanitized_static_values = {}
        for tool in tools:
            if (
                isinstance(tool, ArgumentPropertiesMixin)
                and isinstance(tool, StructuredTool)
                and tool.argument_properties
            ):
                static_args = _resolve_argument_properties(
                    tool.argument_properties, agent_input, tool_name=tool.name
                )
                modified_tool, applied_paths = _apply_static_arguments_to_schema(
                    tool, static_args
                )
                self._processed_tools.append(modified_tool)
                # Only thread args that survived schema modification: paths the
                # schema rejected would fail the synthesized strict validator.
                applied_static_values = {
                    path: sa.value
                    for path, sa in static_args.items()
                    if path in applied_paths
                }
                self._sanitized_static_values[tool.name] = (
                    sanitize_dict_for_serialization(applied_static_values)
                )
            else:
                self._processed_tools.append(tool)

        return self._processed_tools

    def apply_to_response(self, tool_calls: list[ToolCall]) -> None:
        """Applies cached static args to tool calls in-place."""
        if not tool_calls or not self._sanitized_static_values:
            return

        for tool_call in tool_calls:
            static_values = self._sanitized_static_values.get(tool_call["name"])
            if static_values:
                tool_call["args"] = apply_static_args(static_values, tool_call["args"])
