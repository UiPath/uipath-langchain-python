import logging
import re
from typing import Callable, Sequence

from uipath.agent.models.agent import (
    AgentBooleanOperator,
    AgentBooleanRule,
    AgentCustomGuardrail,
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
    StandardRecipient,
)
from uipath.core.guardrails import (
    BooleanRule,
    DeterministicGuardrail,
    NumberRule,
    UniversalRule,
    WordRule,
)
from uipath.platform.guardrails import BaseGuardrail, GuardrailScope

from uipath_langchain.agent.guardrails.actions import (
    BlockAction,
    EscalateAction,
    FilterAction,
    GuardrailAction,
    LogAction,
)
from uipath_langchain.agent.guardrails.utils import _sanitize_selector_tool_names


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
            raise ValueError(f"Unsupported word operator: {operator}")


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
            raise ValueError(f"Unsupported number operator: {operator}")


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
            raise ValueError(f"Unsupported boolean operator: {operator}")


def _convert_agent_rule_to_deterministic(
    agent_rule: AgentWordRule | AgentNumberRule | AgentBooleanRule | UniversalRule,
) -> WordRule | NumberRule | BooleanRule | UniversalRule:
    """Convert an Agent rule to its Deterministic equivalent.

    Args:
        agent_rule: The agent rule to convert.

    Returns:
        The corresponding deterministic rule with a callable function.
    """
    if isinstance(agent_rule, UniversalRule):
        # UniversalRule is already compatible
        return agent_rule

    if isinstance(agent_rule, AgentWordRule):
        return WordRule(
            rule_type="word",
            field_selector=agent_rule.field_selector,
            detects_violation=_create_word_rule_func(
                agent_rule.operator, agent_rule.value
            ),
        )

    if isinstance(agent_rule, AgentNumberRule):
        return NumberRule(
            rule_type="number",
            field_selector=agent_rule.field_selector,
            detects_violation=_create_number_rule_func(
                agent_rule.operator, agent_rule.value
            ),
        )

    if isinstance(agent_rule, AgentBooleanRule):
        return BooleanRule(
            rule_type="boolean",
            field_selector=agent_rule.field_selector,
            detects_violation=_create_boolean_rule_func(
                agent_rule.operator, agent_rule.value
            ),
        )

    raise ValueError(f"Unsupported agent rule type: {type(agent_rule)}")


def _convert_agent_custom_guardrail_to_deterministic(
    guardrail: AgentCustomGuardrail,
) -> DeterministicGuardrail:
    """Convert AgentCustomGuardrail to DeterministicGuardrail.

    Args:
        guardrail: The agent custom guardrail to convert.

    Returns:
        A DeterministicGuardrail with converted rules and sanitized selector.
    """
    converted_rules = [
        _convert_agent_rule_to_deterministic(rule) for rule in guardrail.rules
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
    guardrails: Sequence[AgentGuardrail] | None,
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
                guardrail
            )
            # Validate that DeterministicGuardrails only have TOOL scope
            non_tool_scopes = [
                scope
                for scope in converted_guardrail.selector.scopes
                if scope != GuardrailScope.TOOL
            ]

            if non_tool_scopes:
                raise ValueError(
                    f"Deterministic guardrail '{converted_guardrail.name}' can only be used with TOOL scope. "
                    f"Found invalid scopes: {[scope.name for scope in non_tool_scopes]}. "
                    f"Please configure this guardrail to use only TOOL scope."
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
            if isinstance(action.recipient, StandardRecipient):
                result.append(
                    (
                        converted_guardrail,
                        EscalateAction(
                            app_name=action.app.name,
                            app_folder_path=action.app.folder_name,
                            version=action.app.version,
                            assignee=action.recipient.value,
                        ),
                    )
                )
        elif isinstance(action, AgentGuardrailFilterAction):
            result.append((converted_guardrail, FilterAction(fields=action.fields)))
    return result
