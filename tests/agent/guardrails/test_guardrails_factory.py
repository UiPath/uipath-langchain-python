"""Tests for guardrails_factory.build_guardrails_with_actions."""

import logging
import types
from typing import cast

import pytest
from uipath.agent.models.agent import (  # type: ignore[attr-defined]
    AgentBooleanOperator,
    AgentBooleanRule,
    AgentCustomGuardrail,
    AgentEscalationRecipientType,
    AgentGuardrailActionType,
    AgentGuardrailBlockAction,
    AgentGuardrailEscalateAction,
    AgentGuardrailEscalateActionApp,
    AgentGuardrailFilterAction,
    AgentGuardrailLogAction,
    AgentGuardrailSeverityLevel,
    AgentGuardrailUnknownAction,
    AgentNumberOperator,
    AgentNumberRule,
    AgentWordOperator,
    AgentWordRule,
    FieldReference,
    StandardRecipient,
)
from uipath.agent.models.agent import (
    AgentGuardrail as AgentGuardrailModel,
)
from uipath.core.guardrails import (
    BooleanRule,
    DeterministicGuardrail,
    GuardrailSelector,
    NumberRule,
    WordRule,
)

from uipath_langchain.agent.guardrails.actions.block_action import BlockAction
from uipath_langchain.agent.guardrails.actions.escalate_action import EscalateAction
from uipath_langchain.agent.guardrails.actions.filter_action import FilterAction
from uipath_langchain.agent.guardrails.actions.log_action import LogAction
from uipath_langchain.agent.guardrails.guardrails_factory import (
    _convert_agent_custom_guardrail_to_deterministic,
    _convert_agent_rule_to_deterministic,
    _create_boolean_rule_func,
    _create_number_rule_func,
    _create_word_rule_func,
    build_guardrails_with_actions,
)


class TestGuardrailsFactory:
    def test_none_returns_empty(self) -> None:
        assert build_guardrails_with_actions(None) == []

    def test_empty_list_returns_empty(self) -> None:
        assert build_guardrails_with_actions([]) == []

    def test_block_action_is_mapped_with_reason(self) -> None:
        guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_name",
                selector=GuardrailSelector(),
                action=AgentGuardrailBlockAction(
                    action_type=AgentGuardrailActionType.BLOCK,
                    reason="stop now",
                ),
            ),
        )

        result = build_guardrails_with_actions([guardrail])

        assert len(result) == 1
        gr, action = result[0]
        assert gr is guardrail
        assert isinstance(action, BlockAction)
        assert action.reason == "stop now"

    def test_log_action_is_mapped_with_message_and_severity_level(self) -> None:
        """LOG action is mapped to LogAction with correct message and logging level."""
        guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_log",
                selector=GuardrailSelector(),
                action=AgentGuardrailLogAction(
                    action_type=AgentGuardrailActionType.LOG,
                    message="note this",
                    severity_level=AgentGuardrailSeverityLevel.WARNING,
                ),
            ),
        )

        result = build_guardrails_with_actions([guardrail])

        assert len(result) == 1
        gr, action = result[0]
        assert gr is guardrail
        assert isinstance(action, LogAction)
        assert action.message == "note this"
        assert action.level == logging.WARNING

    def test_unknown_actions_are_ignored(self) -> None:
        """Unknown actions are ignored by the factory."""
        log_guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_1",
                selector=GuardrailSelector(),
                action=AgentGuardrailUnknownAction(
                    action_type=AgentGuardrailActionType.UNKNOWN,
                ),
            ),
        )
        # Mixing UNKNOWN with BLOCK yields only one mapped tuple (BLOCK)
        block_guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_2",
                selector=GuardrailSelector(),
                action=AgentGuardrailBlockAction(
                    action_type=AgentGuardrailActionType.BLOCK,
                    reason="block it",
                ),
            ),
        )
        result = build_guardrails_with_actions([log_guardrail, block_guardrail])
        assert len(result) == 1
        gr, action = result[0]
        assert gr is block_guardrail
        assert isinstance(action, BlockAction)

    def test_escalate_action_is_mapped_with_app_and_recipient(self) -> None:
        """ESCALATE action is mapped to EscalateAction with correct app and recipient."""
        app = AgentGuardrailEscalateActionApp(
            name="EscalationApp",
            folder_name="/TestFolder",
            version=2,
        )
        recipient = StandardRecipient(
            type=AgentEscalationRecipientType.USER_EMAIL,
            value="admin@example.com",
        )
        guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_escalate",
                selector=GuardrailSelector(),
                action=AgentGuardrailEscalateAction(
                    action_type=AgentGuardrailActionType.ESCALATE,
                    app=app,
                    recipient=recipient,
                ),
            ),
        )

        result = build_guardrails_with_actions([guardrail])

        assert len(result) == 1
        gr, action = result[0]
        assert gr is guardrail
        assert isinstance(action, EscalateAction)
        assert action.app_name == "EscalationApp"
        assert action.app_folder_path == "/TestFolder"
        assert action.version == 2
        assert action.assignee == "admin@example.com"

    @pytest.mark.parametrize(
        "scope,scope_lower",
        [
            ("Llm", "llm"),
            ("Agent", "agent"),
        ],
    )
    def test_deterministic_guardrail_with_invalid_scope_raises_value_error(
        self, scope: str, scope_lower: str
    ) -> None:
        """DeterministicGuardrails with LLM or AGENT scope should raise ValueError."""
        guardrail = AgentCustomGuardrail.model_validate(
            {
                "$guardrailType": "custom",
                "id": f"test-{scope_lower}-scope",
                "name": f"test-guardrail-{scope_lower}",
                "description": f"Test guardrail with {scope} scope",
                "enabledForEvals": True,
                "selector": {
                    "$selectorType": "scoped",
                    "scopes": [scope],  # Invalid scope - should be rejected
                    "matchNames": None,
                },
                "rules": [
                    {
                        "$ruleType": "word",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "message.content", "source": "input"}],
                        },
                        "operator": "contains",
                        "value": "forbidden",
                    }
                ],
                "action": {"$actionType": "block", "reason": "test"},
            }
        )

        with pytest.raises(
            ValueError,
            match=rf"Deterministic guardrail 'test-guardrail-{scope_lower}' can only be used with TOOL scope.*Found invalid scopes.*{scope.upper()}",
        ):
            build_guardrails_with_actions([guardrail])

    def test_deterministic_guardrail_with_tool_scope_succeeds(self) -> None:
        """DeterministicGuardrails with TOOL scope should be accepted."""
        guardrail = AgentCustomGuardrail.model_validate(
            {
                "$guardrailType": "custom",
                "id": "test-tool-scope",
                "name": "test-guardrail-tool",
                "description": "Test guardrail with TOOL scope",
                "enabledForEvals": True,
                "selector": {
                    "$selectorType": "scoped",
                    "scopes": ["Tool"],  # TOOL scope - should be accepted
                    "matchNames": ["my_tool"],
                },
                "rules": [
                    {
                        "$ruleType": "word",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "message.content", "source": "input"}],
                        },
                        "operator": "contains",
                        "value": "forbidden",
                    }
                ],
                "action": {"$actionType": "block", "reason": "test"},
            }
        )

        result = build_guardrails_with_actions([guardrail])

        assert len(result) == 1
        converted_guardrail, action = result[0]
        assert isinstance(converted_guardrail, DeterministicGuardrail)
        assert converted_guardrail.name == "test-guardrail-tool"
        assert isinstance(action, BlockAction)

    def test_deterministic_guardrail_with_mixed_scopes_raises_value_error(self) -> None:
        """DeterministicGuardrails with mixed scopes including non-TOOL should raise ValueError."""
        guardrail = AgentCustomGuardrail.model_validate(
            {
                "$guardrailType": "custom",
                "id": "test-mixed-scope",
                "name": "test-guardrail-mixed",
                "description": "Test guardrail with mixed scopes",
                "enabledForEvals": True,
                "selector": {
                    "$selectorType": "scoped",
                    "scopes": ["Tool", "Llm"],  # Mixed scopes - should be rejected
                    "matchNames": ["my_tool"],
                },
                "rules": [
                    {
                        "$ruleType": "word",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "message.content", "source": "input"}],
                        },
                        "operator": "contains",
                        "value": "forbidden",
                    }
                ],
                "action": {"$actionType": "block", "reason": "test"},
            }
        )

        with pytest.raises(
            ValueError,
            match=r"Deterministic guardrail 'test-guardrail-mixed' can only be used with TOOL scope.*Found invalid scopes.*LLM",
        ):
            build_guardrails_with_actions([guardrail])

    def test_filter_action_is_mapped_with_fields(self) -> None:
        """FILTER action is mapped to FilterAction with correct fields."""
        fields = [
            FieldReference(path="data.password", source="input"),
            FieldReference(path="data.ssn", source="input"),
        ]
        guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_filter",
                selector=GuardrailSelector(),
                action=AgentGuardrailFilterAction(
                    action_type=AgentGuardrailActionType.FILTER,
                    fields=fields,
                ),
            ),
        )

        result = build_guardrails_with_actions([guardrail])

        assert len(result) == 1
        gr, action = result[0]
        assert gr is guardrail
        assert isinstance(action, FilterAction)
        assert action.fields == fields
        assert len(action.fields) == 2
        assert action.fields[0].path == "data.password"
        assert action.fields[1].path == "data.ssn"

    def test_filter_action_with_custom_guardrail(self) -> None:
        """FILTER action works with AgentCustomGuardrail and converts to DeterministicGuardrail."""
        agent_guardrail = AgentCustomGuardrail.model_validate(
            {
                "$guardrailType": "custom",
                "id": "filter-test-id",
                "name": "filter-guardrail",
                "description": "Filter sensitive data",
                "enabledForEvals": True,
                "selector": {"$selectorType": "all"},
                "rules": [
                    {
                        "$ruleType": "word",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "data.secret", "source": "input"}],
                        },
                        "operator": "isNotEmpty",
                        "value": None,
                    }
                ],
                "action": {
                    "$actionType": "filter",
                    "fields": [{"path": "data.secret", "source": "input"}],
                },
            }
        )

        result = build_guardrails_with_actions([agent_guardrail])

        assert len(result) == 1
        gr, action = result[0]
        # Guardrail should be converted to DeterministicGuardrail
        assert isinstance(gr, DeterministicGuardrail)
        assert gr.name == "filter-guardrail"
        assert isinstance(action, FilterAction)
        assert len(action.fields) == 1

    def test_multiple_guardrails_with_different_actions_including_filter(self) -> None:
        """Multiple guardrails with different action types including FILTER are all mapped."""
        fields = [FieldReference(path="data.token", source="input")]

        block_guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="block_gr",
                selector=GuardrailSelector(),
                action=AgentGuardrailBlockAction(
                    action_type=AgentGuardrailActionType.BLOCK,
                    reason="blocked",
                ),
            ),
        )

        filter_guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="filter_gr",
                selector=GuardrailSelector(),
                action=AgentGuardrailFilterAction(
                    action_type=AgentGuardrailActionType.FILTER,
                    fields=fields,
                ),
            ),
        )

        log_guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="log_gr",
                selector=GuardrailSelector(),
                action=AgentGuardrailLogAction(
                    action_type=AgentGuardrailActionType.LOG,
                    message="logged",
                    severity_level=AgentGuardrailSeverityLevel.INFO,
                ),
            ),
        )

        result = build_guardrails_with_actions(
            [block_guardrail, filter_guardrail, log_guardrail]
        )

        assert len(result) == 3

        # Check block action
        assert isinstance(result[0][1], BlockAction)
        assert result[0][1].reason == "blocked"

        # Check filter action
        assert isinstance(result[1][1], FilterAction)
        assert result[1][1].fields == fields
        assert len(result[1][1].fields) == 1
        assert result[1][1].fields[0].path == "data.token"

        # Check log action
        assert isinstance(result[2][1], LogAction)
        assert result[2][1].message == "logged"


class TestCreateWordRuleFunc:
    """Tests for _create_word_rule_func."""

    def test_contains_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.CONTAINS, "test")
        assert func("this is a test") is True
        assert func("this is a TEST") is True  # case-insensitive
        assert func("this is a TeSt") is True  # case-insensitive
        assert func("no match") is False

    def test_does_not_contain_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.DOES_NOT_CONTAIN, "test")
        assert func("no match") is True
        assert func("this is a test") is False
        assert func("this is a TEST") is False  # case-insensitive
        assert func("this is a TeSt") is False  # case-insensitive

    def test_equals_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.EQUALS, "exact")
        assert func("exact") is True
        assert func("not exact") is False

    def test_does_not_equal_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.DOES_NOT_EQUAL, "exact")
        assert func("different") is True
        assert func("exact") is False

    def test_starts_with_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.STARTS_WITH, "hello")
        assert func("hello world") is True
        assert func("world hello") is False

    def test_does_not_start_with_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.DOES_NOT_START_WITH, "hello")
        assert func("world hello") is True
        assert func("hello world") is False

    def test_ends_with_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.ENDS_WITH, "world")
        assert func("hello world") is True
        assert func("world hello") is False

    def test_does_not_end_with_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.DOES_NOT_END_WITH, "world")
        assert func("world hello") is True
        assert func("hello world") is False

    def test_is_empty_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.IS_EMPTY, None)
        assert func("") is True
        assert func("not empty") is False

    def test_is_not_empty_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.IS_NOT_EMPTY, None)
        assert func("not empty") is True
        assert func("") is False

    def test_matches_regex_operator(self) -> None:
        func = _create_word_rule_func(AgentWordOperator.MATCHES_REGEX, r"^\d{3}$")
        assert func("123") is True
        assert func("abc") is False
        assert func("1234") is False

    def test_matches_regex_with_none_value_raises_assertion_error(self) -> None:
        with pytest.raises(
            AssertionError, match="value cannot be None for MATCHES_REGEX operator"
        ):
            _create_word_rule_func(AgentWordOperator.MATCHES_REGEX, None)

    def test_contains_with_none_value_raises_assertion_error(self) -> None:
        with pytest.raises(
            AssertionError, match="value cannot be None for CONTAINS operator"
        ):
            _create_word_rule_func(AgentWordOperator.CONTAINS, None)

    def test_starts_with_none_value_raises_assertion_error(self) -> None:
        with pytest.raises(
            AssertionError, match="value cannot be None for STARTS_WITH operator"
        ):
            _create_word_rule_func(AgentWordOperator.STARTS_WITH, None)

    def test_unsupported_operator_raises_value_error(self) -> None:
        # Create a mock operator that's not in the supported list
        with pytest.raises(ValueError, match="Unsupported word operator"):
            _create_word_rule_func(cast(AgentWordOperator, "INVALID"), "value")


class TestCreateNumberRuleFunc:
    """Tests for _create_number_rule_func."""

    def test_equals_operator(self) -> None:
        func = _create_number_rule_func(AgentNumberOperator.EQUALS, 10.0)
        assert func(10.0) is True
        assert func(5.0) is False

    def test_does_not_equal_operator(self) -> None:
        func = _create_number_rule_func(AgentNumberOperator.DOES_NOT_EQUAL, 10.0)
        assert func(5.0) is True
        assert func(10.0) is False

    def test_greater_than_operator(self) -> None:
        func = _create_number_rule_func(AgentNumberOperator.GREATER_THAN, 10.0)
        assert func(15.0) is True
        assert func(10.0) is False
        assert func(5.0) is False

    def test_greater_than_or_equal_operator(self) -> None:
        func = _create_number_rule_func(AgentNumberOperator.GREATER_THAN_OR_EQUAL, 10.0)
        assert func(15.0) is True
        assert func(10.0) is True
        assert func(5.0) is False

    def test_less_than_operator(self) -> None:
        func = _create_number_rule_func(AgentNumberOperator.LESS_THAN, 10.0)
        assert func(5.0) is True
        assert func(10.0) is False
        assert func(15.0) is False

    def test_less_than_or_equal_operator(self) -> None:
        func = _create_number_rule_func(AgentNumberOperator.LESS_THAN_OR_EQUAL, 10.0)
        assert func(5.0) is True
        assert func(10.0) is True
        assert func(15.0) is False

    def test_unsupported_operator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported number operator"):
            _create_number_rule_func(cast(AgentNumberOperator, "INVALID"), 10.0)


class TestCreateBooleanRuleFunc:
    """Tests for _create_boolean_rule_func."""

    def test_equals_operator_true(self) -> None:
        func = _create_boolean_rule_func(AgentBooleanOperator.EQUALS, True)
        assert func(True) is True
        assert func(False) is False

    def test_equals_operator_false(self) -> None:
        func = _create_boolean_rule_func(AgentBooleanOperator.EQUALS, False)
        assert func(False) is True
        assert func(True) is False

    def test_unsupported_operator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported boolean operator"):
            _create_boolean_rule_func(cast(AgentBooleanOperator, "INVALID"), True)


class TestConvertAgentRuleToDeterministic:
    """Tests for _convert_agent_rule_to_deterministic."""

    def test_convert_word_rule(self) -> None:
        # Create AgentWordRule using model_validate to handle the $selectorType field
        agent_rule = AgentWordRule.model_validate(
            {
                "$ruleType": "word",
                "fieldSelector": {
                    "$selectorType": "specific",
                    "fields": [{"path": "message.content", "source": "input"}],
                },
                "operator": "contains",
                "value": "test",
            }
        )
        result = _convert_agent_rule_to_deterministic(agent_rule)

        assert isinstance(result, WordRule)
        assert result.rule_type == "word"
        # field_selector is now the actual selector object, not a string
        assert result.detects_violation("this is a test") is True
        assert result.detects_violation("no match") is False

    def test_convert_number_rule(self) -> None:
        agent_rule = AgentNumberRule.model_validate(
            {
                "$ruleType": "number",
                "fieldSelector": {
                    "$selectorType": "specific",
                    "fields": [{"path": "data.count", "source": "input"}],
                },
                "operator": "greaterThan",
                "value": 10.0,
            }
        )
        result = _convert_agent_rule_to_deterministic(agent_rule)

        assert isinstance(result, NumberRule)
        assert result.rule_type == "number"
        assert result.detects_violation(15.0) is True
        assert result.detects_violation(5.0) is False

    def test_convert_boolean_rule(self) -> None:
        agent_rule = AgentBooleanRule.model_validate(
            {
                "$ruleType": "boolean",
                "fieldSelector": {
                    "$selectorType": "specific",
                    "fields": [{"path": "data.is_active", "source": "input"}],
                },
                "operator": "equals",
                "value": True,
            }
        )
        result = _convert_agent_rule_to_deterministic(agent_rule)

        assert isinstance(result, BooleanRule)
        assert result.rule_type == "boolean"
        assert result.detects_violation(True) is True
        assert result.detects_violation(False) is False

    def test_unsupported_rule_type_raises_value_error(self) -> None:
        # Create a mock rule that's not a supported type
        invalid_rule = cast(AgentWordRule, types.SimpleNamespace())
        with pytest.raises(ValueError, match="Unsupported agent rule type"):
            _convert_agent_rule_to_deterministic(invalid_rule)


class TestConvertAgentCustomGuardrailToDeterministic:
    """Tests for _convert_agent_custom_guardrail_to_deterministic."""

    def test_convert_custom_guardrail_with_word_rules(self) -> None:
        agent_guardrail = AgentCustomGuardrail.model_validate(
            {
                "$guardrailType": "custom",
                "id": "test-id",
                "name": "test-guardrail",
                "description": "Test guardrail description",
                "enabledForEvals": True,
                "selector": {"$selectorType": "all"},
                "rules": [
                    {
                        "$ruleType": "word",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "message.content", "source": "input"}],
                        },
                        "operator": "contains",
                        "value": "forbidden",
                    },
                    {
                        "$ruleType": "word",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "message.content", "source": "input"}],
                        },
                        "operator": "startsWith",
                        "value": "admin",
                    },
                ],
                "action": {"$actionType": "block", "reason": "test"},
            }
        )

        result = _convert_agent_custom_guardrail_to_deterministic(agent_guardrail)

        assert isinstance(result, DeterministicGuardrail)
        assert result.id == "test-id"
        assert result.name == "test-guardrail"
        assert result.description == "Test guardrail description"
        assert result.enabled_for_evals is True
        assert result.guardrail_type == "custom"
        assert len(result.rules) == 2
        assert isinstance(result.rules[0], WordRule)
        assert isinstance(result.rules[1], WordRule)

    def test_convert_custom_guardrail_with_mixed_rules(self) -> None:
        agent_guardrail = AgentCustomGuardrail.model_validate(
            {
                "$guardrailType": "custom",
                "id": "mixed-id",
                "name": "mixed-guardrail",
                "description": "Mixed rules guardrail",
                "enabledForEvals": False,
                "selector": {"$selectorType": "all"},
                "rules": [
                    {
                        "$ruleType": "word",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "message.content", "source": "input"}],
                        },
                        "operator": "contains",
                        "value": "test",
                    },
                    {
                        "$ruleType": "number",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "data.count", "source": "input"}],
                        },
                        "operator": "greaterThan",
                        "value": 5.0,
                    },
                    {
                        "$ruleType": "boolean",
                        "fieldSelector": {
                            "$selectorType": "specific",
                            "fields": [{"path": "data.is_active", "source": "input"}],
                        },
                        "operator": "equals",
                        "value": True,
                    },
                ],
                "action": {"$actionType": "block", "reason": "test"},
            }
        )

        result = _convert_agent_custom_guardrail_to_deterministic(agent_guardrail)

        assert isinstance(result, DeterministicGuardrail)
        assert len(result.rules) == 3
        assert isinstance(result.rules[0], WordRule)
        assert isinstance(result.rules[1], NumberRule)
        assert isinstance(result.rules[2], BooleanRule)

    def test_convert_custom_guardrail_with_empty_rules(self) -> None:
        agent_guardrail = AgentCustomGuardrail.model_validate(
            {
                "$guardrailType": "custom",
                "id": "empty-id",
                "name": "empty-guardrail",
                "description": "Empty rules guardrail",
                "enabledForEvals": True,
                "selector": {"$selectorType": "all"},
                "rules": [],
                "action": {"$actionType": "block", "reason": "test"},
            }
        )

        result = _convert_agent_custom_guardrail_to_deterministic(agent_guardrail)

        assert isinstance(result, DeterministicGuardrail)
        assert len(result.rules) == 0
