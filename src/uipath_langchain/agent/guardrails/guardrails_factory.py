import logging
import re
from typing import Callable, Sequence

from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentAllFieldsSelector,
    AgentBooleanOperator,
    AgentBooleanRule,
    AgentCustomGuardrail,
    AgentFieldSelector,
    AgentGuardrail,
    AgentGuardrailBlockAction,
    AgentGuardrailEscalateAction,
    AgentGuardrailFilterAction,
    AgentGuardrailLogAction,
    AgentGuardrailSeverityLevel,
    AgentNumberOperator,
    AgentNumberRule,
    AgentUnknownGuardrail,
    AgentWordOperator,
    AgentWordRule,
)
from uipath.core.guardrails import (
    AllFieldsSelector,
    BooleanRule,
    DeterministicGuardrail,
    FieldSelector,
    FieldSource,
    NumberRule,
    SpecificFieldsSelector,
    UniversalRule,
    WordRule,
)
from uipath.platform.guardrails import BaseGuardrail, GuardrailScope
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.agent.guardrails.actions import (
    BlockAction,
    EscalateAction,
    FilterAction,
    GuardrailAction,
    LogAction,
)
from uipath_langchain.agent.guardrails.utils import _sanitize_selector_tool_names


def _has_schema(tool: BaseTool, attribute_name: str) -> bool:
    """Check if a tool has a non-empty schema for the given attribute.

    Args:
        tool: The tool to check.
        attribute_name: The name of the attribute to check (e.g., 'args_schema', 'output_type').

    Returns:
        True if the tool has a non-empty schema with properties, False otherwise.
    """
    # Check if tool has the attribute
    if not hasattr(tool, attribute_name):
        return False

    schema_obj = getattr(tool, attribute_name, None)
    if schema_obj is None:
        return False

    # Get the JSON schema from the schema object
    try:
        if hasattr(schema_obj, "model_json_schema"):
            schema = schema_obj.model_json_schema()
            # Check if schema has properties
            properties = schema.get("properties", {})
            return bool(properties)
    except Exception:
        pass

    return False


def _tool_has_input_schema(tool: BaseTool) -> bool:
    """Check if a tool has a non-empty input schema.

    Args:
        tool: The tool to check.

    Returns:
        True if the tool has a non-empty input schema, False otherwise.
    """
    return _has_schema(tool, "args_schema")


def _tool_has_output_schema(tool: BaseTool) -> bool:
    """Check if a tool has a non-empty output schema.

    Args:
        tool: The tool to check.

    Returns:
        True if the tool has a non-empty output schema, False otherwise.
    """
    return _has_schema(tool, "output_type")


def _assert_value_not_none(value: str | None, operator: AgentWordOperator) -> str:
    """Assert value is not None and return as string."""
    assert value is not None, f"value cannot be None for {operator.name} operator"
    return value


def _create_word_rule_func(
    operator: AgentWordOperator, value: str | None
) -> Callable[[str], bool]:
    """Create a callable function from AgentWordOperator and value.

    Args:
        operator: The word operator to convert.
        value: The value to compare against (may be None for isEmpty/isNotEmpty).

    Returns:
        A callable that takes a string and returns a boolean.
    """
    match operator:
        case AgentWordOperator.CONTAINS:
            val = _assert_value_not_none(value, operator)
            return lambda s: val.lower() in s.lower()
        case AgentWordOperator.DOES_NOT_CONTAIN:
            val = _assert_value_not_none(value, operator)
            return lambda s: val.lower() not in s.lower()
        case AgentWordOperator.EQUALS:
            val = _assert_value_not_none(value, operator)
            return lambda s: s == val
        case AgentWordOperator.DOES_NOT_EQUAL:
            val = _assert_value_not_none(value, operator)
            return lambda s: s != val
        case AgentWordOperator.STARTS_WITH:
            val = _assert_value_not_none(value, operator)
            return lambda s: s.startswith(val)
        case AgentWordOperator.DOES_NOT_START_WITH:
            val = _assert_value_not_none(value, operator)
            return lambda s: not s.startswith(val)
        case AgentWordOperator.ENDS_WITH:
            val = _assert_value_not_none(value, operator)
            return lambda s: s.endswith(val)
        case AgentWordOperator.DOES_NOT_END_WITH:
            val = _assert_value_not_none(value, operator)
            return lambda s: not s.endswith(val)
        case AgentWordOperator.IS_EMPTY:
            return lambda s: len(s) == 0
        case AgentWordOperator.IS_NOT_EMPTY:
            return lambda s: len(s) > 0
        case AgentWordOperator.MATCHES_REGEX:
            val = _assert_value_not_none(value, operator)
            pattern = re.compile(val)
            return lambda s: bool(pattern.match(s))
        case _:
            raise AgentStartupError(
                code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
                title="Unsupported word operator",
                detail=f"Unsupported word operator: {operator}. Please review your guardrail config.",
                category=UiPathErrorCategory.USER,
            )


def _build_field_selector_description(field_selector: AgentFieldSelector) -> str:
    """Build a human-readable selector description for field selector.

    Args:
        field_selector: The field selector describing which fields this rule applies to.

    Returns:
        A string describing the selector, using:
        - \"All\" for `AgentAllFieldsSelector`
        - Comma-separated field paths for `SpecificFieldsSelector`
        - ``str(field_selector)`` as a fallback.
    """
    if isinstance(field_selector, AgentAllFieldsSelector):
        return "All fields"
    if isinstance(field_selector, SpecificFieldsSelector):
        field_paths = [field.path for field in field_selector.fields]
        return ", ".join(field_paths)
    return str(field_selector)


def _build_rule_description(
    operator: AgentWordOperator | AgentNumberOperator | AgentBooleanOperator,
    value: str | float | bool | None,
    field_selector: AgentFieldSelector,
) -> str:
    """Build the full human-readable description for a word rule.

    Args:
        operator: The word operator to describe.
        value: The comparison value, if applicable for the operator.
        field_selector: The field selector describing which fields this rule applies to.

    Returns:
        A string describing the rule, combining selector, operator, and value.
    """
    selector_description = _build_field_selector_description(field_selector)

    if operator in {
        AgentWordOperator.CONTAINS,
        AgentWordOperator.DOES_NOT_CONTAIN,
        AgentWordOperator.EQUALS,
        AgentWordOperator.DOES_NOT_EQUAL,
        AgentWordOperator.STARTS_WITH,
        AgentWordOperator.DOES_NOT_START_WITH,
        AgentWordOperator.ENDS_WITH,
        AgentWordOperator.DOES_NOT_END_WITH,
        AgentWordOperator.MATCHES_REGEX,
        AgentNumberOperator.EQUALS,
        AgentNumberOperator.DOES_NOT_EQUAL,
        AgentNumberOperator.GREATER_THAN,
        AgentNumberOperator.GREATER_THAN_OR_EQUAL,
        AgentNumberOperator.LESS_THAN,
        AgentNumberOperator.LESS_THAN_OR_EQUAL,
        AgentBooleanOperator.EQUALS,
    }:
        return f"{selector_description} {operator.value} {value!r}"

    if operator in {
        AgentWordOperator.IS_EMPTY,
        AgentWordOperator.IS_NOT_EMPTY,
    }:
        return f"{selector_description} {operator.value}"

    raise AgentStartupError(
        code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
        title="Unsupported word operator",
        detail=f"Unsupported word operator: {operator}. Please review your guardrail config.",
        category=UiPathErrorCategory.USER,
    )


def _create_number_rule_func(
    operator: AgentNumberOperator, value: float
) -> Callable[[float], bool]:
    """Create a callable function from AgentNumberOperator and value.

    Args:
        operator: The number operator to convert.
        value: The value to compare against.

    Returns:
        A callable that takes a float and returns a boolean.
    """
    match operator:
        case AgentNumberOperator.EQUALS:
            return lambda n: n == value
        case AgentNumberOperator.DOES_NOT_EQUAL:
            return lambda n: n != value
        case AgentNumberOperator.GREATER_THAN:
            return lambda n: n > value
        case AgentNumberOperator.GREATER_THAN_OR_EQUAL:
            return lambda n: n >= value
        case AgentNumberOperator.LESS_THAN:
            return lambda n: n < value
        case AgentNumberOperator.LESS_THAN_OR_EQUAL:
            return lambda n: n <= value
        case _:
            raise AgentStartupError(
                code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
                title="Unsupported number operator",
                detail=f"Unsupported number operator: {operator}. Please review your guardrail config.",
                category=UiPathErrorCategory.USER,
            )


def _create_boolean_rule_func(
    operator: AgentBooleanOperator, value: bool
) -> Callable[[bool], bool]:
    """Create a callable function from AgentBooleanOperator and value.

    Args:
        operator: The boolean operator to convert.
        value: The value to compare against.

    Returns:
        A callable that takes a boolean and returns a boolean.
    """
    match operator:
        case AgentBooleanOperator.EQUALS:
            return lambda b: b == value
        case _:
            raise AgentStartupError(
                code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
                title="Unsupported boolean operator",
                detail=f"Unsupported boolean operator: {operator}. Please review your guardrail config.",
                category=UiPathErrorCategory.USER,
            )


def _compute_field_sources_for_guardrail(
    guardrail: AgentCustomGuardrail,
    tools: list[BaseTool],
) -> list[FieldSource]:
    """Compute field sources based on tool input and output schemas.

    Args:
        guardrail: The guardrail to extract tool information from.
        tools: The list of tools available to the agent.

    Returns:
        List of field sources based on tool schema when matching tool is found:
        - [] (empty) if tool has neither input nor output schema
        - [INPUT] if tool has only input schema
        - [OUTPUT] if tool has only output schema
        - [INPUT, OUTPUT] if tool has both input and output schemas

    Raises:
        AgentStartupError: If match_names is not specified/empty, or if the specified tool
                   is not found in the provided tools list.
    """
    field_sources = []

    # Deterministic guardrails have one single tool
    if guardrail.selector.match_names and len(guardrail.selector.match_names) > 0:
        matching_tool = next(
            (t for t in tools if t.name == guardrail.selector.match_names[0]), None
        )

        if matching_tool:
            has_input = _tool_has_input_schema(matching_tool)
            has_output = _tool_has_output_schema(matching_tool)

            # Start with empty list and add sources based on what tool has
            if has_input:
                field_sources.append(FieldSource.INPUT)
            if has_output:
                field_sources.append(FieldSource.OUTPUT)

            return field_sources

    # If we reach here, either match_names is not specified/empty, or no matching tool was found
    tool_name: str | None = (
        guardrail.selector.match_names[0] if guardrail.selector.match_names else None
    )
    raise AgentStartupError(
        code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
        title="Invalid guardrail match_names",
        detail=f"Guardrail '{guardrail.name}' requires a valid match_names with at least one tool. "
        f"Tool '{tool_name}' not found in available tools. Please verify the tool name in your guardrail config."
        if tool_name
        else f"Guardrail '{guardrail.name}' has match_names empty or not specified. Please add at least one tool name to match_names.",
        category=UiPathErrorCategory.USER,
    )


def _convert_agent_field_selector_to_deterministic(
    agent_field_selector: AgentFieldSelector,
    guardrail: AgentCustomGuardrail,
    tools: list[BaseTool],
) -> FieldSelector:
    """Convert an Agent field selector to its Deterministic equivalent.

    Args:
        agent_field_selector: The agent field selector to convert.
        guardrail: The guardrail to extract tool information from.
        tools: The list of tools available to the agent.

    Returns:
        The corresponding deterministic field selector.
    """
    if isinstance(agent_field_selector, AgentAllFieldsSelector):
        field_sources = _compute_field_sources_for_guardrail(guardrail, tools)
        return AllFieldsSelector(
            selector_type=agent_field_selector.selector_type,
            sources=field_sources,
        )
    elif isinstance(agent_field_selector, SpecificFieldsSelector):
        # SpecificFieldsSelector is already compatible
        return agent_field_selector
    else:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
            title="Unsupported field selector type",
            detail=f"Unsupported agent field selector type: {type(agent_field_selector)}. Please review your guardrail config.",
            category=UiPathErrorCategory.USER,
        )


def _convert_agent_rule_to_deterministic(
    agent_rule: AgentWordRule | AgentNumberRule | AgentBooleanRule | UniversalRule,
    guardrail: AgentCustomGuardrail,
    tools: list[BaseTool],
) -> WordRule | NumberRule | BooleanRule | UniversalRule:
    """Convert an Agent rule to its Deterministic equivalent.

    Args:
        agent_rule: The agent rule to convert.
        guardrail: The parent guardrail (for accessing selector information).
        tools: The list of tools available to the agent.

    Returns:
        The corresponding deterministic rule with a callable function.
    """
    if isinstance(agent_rule, UniversalRule):
        # UniversalRule is already compatible
        agent_rule.rule_description = (
            f"Always enforce the guardrail on {agent_rule.apply_to.value}"
        )
        return agent_rule

    if isinstance(agent_rule, AgentWordRule):
        return WordRule(
            rule_type="word",
            field_selector=_convert_agent_field_selector_to_deterministic(
                agent_rule.field_selector, guardrail, tools
            ),
            detects_violation=_create_word_rule_func(
                agent_rule.operator, agent_rule.value
            ),
            rule_description=_build_rule_description(
                agent_rule.operator, agent_rule.value, agent_rule.field_selector
            ),
        )

    if isinstance(agent_rule, AgentNumberRule):
        return NumberRule(
            rule_type="number",
            field_selector=_convert_agent_field_selector_to_deterministic(
                agent_rule.field_selector, guardrail, tools
            ),
            detects_violation=_create_number_rule_func(
                agent_rule.operator, agent_rule.value
            ),
            rule_description=_build_rule_description(
                agent_rule.operator, agent_rule.value, agent_rule.field_selector
            ),
        )

    if isinstance(agent_rule, AgentBooleanRule):
        return BooleanRule(
            rule_type="boolean",
            field_selector=_convert_agent_field_selector_to_deterministic(
                agent_rule.field_selector, guardrail, tools
            ),
            detects_violation=_create_boolean_rule_func(
                agent_rule.operator, agent_rule.value
            ),
            rule_description=_build_rule_description(
                agent_rule.operator, agent_rule.value, agent_rule.field_selector
            ),
        )

    raise AgentStartupError(
        code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
        title="Unsupported rule type",
        detail=f"Unsupported agent rule type: {type(agent_rule)}. Please review your guardrail config.",
        category=UiPathErrorCategory.USER,
    )


def _convert_agent_custom_guardrail_to_deterministic(
    guardrail: AgentCustomGuardrail,
    tools: list[BaseTool],
) -> DeterministicGuardrail:
    """Convert AgentCustomGuardrail to DeterministicGuardrail.

    Args:
        guardrail: The agent custom guardrail to convert.
        tools: The list of tools available to the agent.

    Returns:
        A DeterministicGuardrail with converted rules and sanitized selector.
    """
    converted_rules = [
        _convert_agent_rule_to_deterministic(rule, guardrail, tools)
        for rule in guardrail.rules
    ]

    # Sanitize tool names in selector for Tool scope guardrails
    sanitized_selector = _sanitize_selector_tool_names(guardrail.selector)

    return DeterministicGuardrail(
        id=guardrail.id,
        name=guardrail.name,
        description=guardrail.description,
        enabled_for_evals=guardrail.enabled_for_evals,
        selector=sanitized_selector,
        guardrail_type="custom",
        rules=converted_rules,
    )


def build_guardrails_with_actions(
    guardrails: Sequence[AgentGuardrail] | None, tools: list[BaseTool]
) -> list[tuple[BaseGuardrail, GuardrailAction]]:
    """Build a list of (guardrail, action) tuples from model definitions.

    Args:
        guardrails: Sequence of guardrail model objects or None.

    Returns:
        A list of tuples pairing each supported guardrail with its executable action.
    """
    if not guardrails:
        return []

    result: list[tuple[BaseGuardrail, GuardrailAction]] = []
    for guardrail in guardrails:
        if isinstance(guardrail, AgentUnknownGuardrail):
            continue

        converted_guardrail: BaseGuardrail
        if isinstance(guardrail, AgentCustomGuardrail):
            converted_guardrail = _convert_agent_custom_guardrail_to_deterministic(
                guardrail, tools
            )
            # Validate that DeterministicGuardrails only have TOOL scope
            non_tool_scopes = [
                scope
                for scope in converted_guardrail.selector.scopes
                if scope != GuardrailScope.TOOL
            ]

            if non_tool_scopes:
                raise AgentStartupError(
                    code=AgentStartupErrorCode.INVALID_GUARDRAIL_CONFIG,
                    title="Invalid guardrail scope",
                    detail=f"Deterministic guardrail '{converted_guardrail.name}' can only be used with TOOL scope. "
                    f"Found invalid scopes: {[scope.name for scope in non_tool_scopes]}. "
                    f"Please configure this guardrail to use only TOOL scope.",
                    category=UiPathErrorCategory.USER,
                )
        else:
            converted_guardrail = guardrail
            _sanitize_selector_tool_names(converted_guardrail.selector)

        action = guardrail.action

        if isinstance(action, AgentGuardrailBlockAction):
            result.append((converted_guardrail, BlockAction(action.reason)))
        elif isinstance(action, AgentGuardrailLogAction):
            severity_level_map = {
                AgentGuardrailSeverityLevel.ERROR: logging.ERROR,
                AgentGuardrailSeverityLevel.WARNING: logging.WARNING,
                AgentGuardrailSeverityLevel.INFO: logging.INFO,
            }
            level = severity_level_map.get(action.severity_level, logging.INFO)
            result.append(
                (
                    converted_guardrail,
                    LogAction(message=action.message, level=level),
                )
            )
        elif isinstance(action, AgentGuardrailEscalateAction):
            result.append(
                (
                    converted_guardrail,
                    EscalateAction(
                        app_name=action.app.name,
                        app_folder_path=action.app.folder_name,
                        version=action.app.version,
                        recipient=action.recipient,
                    ),
                )
            )
        elif isinstance(action, AgentGuardrailFilterAction):
            result.append((converted_guardrail, FilterAction(fields=action.fields)))
    return result
