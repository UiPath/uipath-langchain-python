"""Tests for ``UiPathByoGuardrailMiddleware`` (Bring Your Own Guardrail).

BYOG guardrails reference an admin-created configuration by validator name
(plus an optional Integration Service connection id). These tests pin:
- the ``BuiltInValidatorGuardrail`` the middleware constructs (``validator_type
  == "byo"`` sentinel + ``byoValidatorName``/``byoConnectionId`` aliases),
- constructor validation,
- hook wiring parity with the other built-in middlewares (scopes x stage),
- that the evaluation path forwards the BYO guardrail to the service and that
  ``BlockAction`` raises on ``VALIDATION_FAILED``.
"""

from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock

import pytest
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import GuardrailScope
from uipath.platform.guardrails.decorators import BlockAction, LogAction
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException
from uipath.platform.guardrails.guardrails import (
    BYO_VALIDATOR_TYPE,
    NumberParameterValue,
)

from uipath_langchain.guardrails.enums import GuardrailExecutionStage
from uipath_langchain.guardrails.middlewares import UiPathByoGuardrailMiddleware

_LOG = LogAction()
_BLOCK = BlockAction()

_VALIDATOR_NAME = "my-harmful-content-guardrail"
_CONNECTION_ID = "my-byog-guardrail-connection"


def _hook_names(middleware: Iterable[Any]) -> list[str]:
    """Return the ``name`` attribute of each AgentMiddleware instance."""
    return [inst.name for inst in middleware]


def _byo(**overrides: Any) -> UiPathByoGuardrailMiddleware:
    kwargs: dict[str, Any] = {
        "validator_name": _VALIDATOR_NAME,
        "scopes": [GuardrailScope.AGENT],
        "action": _LOG,
    }
    kwargs.update(overrides)
    return UiPathByoGuardrailMiddleware(**kwargs)


class TestByoGuardrailConstruction:
    """The constructed guardrail carries the BYO sentinel and reference fields."""

    def test_guardrail_uses_byo_sentinel_validator_type(self) -> None:
        middleware = _byo()
        assert middleware._guardrail.validator_type == BYO_VALIDATOR_TYPE

    def test_guardrail_carries_validator_name_and_connection_id(self) -> None:
        middleware = _byo(connection_id=_CONNECTION_ID)
        assert middleware._guardrail.byo_validator_name == _VALIDATOR_NAME
        assert middleware._guardrail.byo_connection_id == _CONNECTION_ID

    def test_connection_id_defaults_to_none(self) -> None:
        middleware = _byo()
        assert middleware._guardrail.byo_connection_id is None

    def test_aliases_serialize_for_the_wire(self) -> None:
        middleware = _byo(connection_id=_CONNECTION_ID)
        dumped = middleware._guardrail.model_dump(by_alias=True)
        assert dumped["validatorType"] == "byo"
        assert dumped["byoValidatorName"] == _VALIDATOR_NAME
        assert dumped["byoConnectionId"] == _CONNECTION_ID
        assert dumped["$guardrailType"] == "builtInValidator"

    def test_validator_parameters_pass_through(self) -> None:
        parameter = NumberParameterValue(
            parameter_type="number", id="threshold", value=0.7
        )
        middleware = _byo(validator_parameters=[parameter])
        assert middleware._guardrail.validator_parameters == [parameter]

    def test_validator_parameters_default_empty(self) -> None:
        middleware = _byo()
        assert middleware._guardrail.validator_parameters == []

    def test_default_name_and_description_include_validator_name(self) -> None:
        middleware = _byo()
        assert _VALIDATOR_NAME in middleware._guardrail.name
        assert middleware._guardrail.description is not None
        assert _VALIDATOR_NAME in middleware._guardrail.description

    def test_explicit_name_and_description_win(self) -> None:
        middleware = _byo(name="My Guardrail", description="My description")
        assert middleware._guardrail.name == "My Guardrail"
        assert middleware._guardrail.description == "My description"

    def test_tool_scope_sets_selector_match_names(self) -> None:
        middleware = _byo(
            scopes=[GuardrailScope.TOOL],
            tools=["my_tool"],
        )
        assert middleware._guardrail.selector is not None
        assert middleware._guardrail.selector.match_names == ["my_tool"]


class TestByoGuardrailValidation:
    """Constructor validation errors."""

    def test_empty_validator_name_raises(self) -> None:
        with pytest.raises(ValueError, match="validator_name"):
            _byo(validator_name="")

    def test_whitespace_validator_name_raises(self) -> None:
        with pytest.raises(ValueError, match="validator_name"):
            _byo(validator_name="   ")

    def test_empty_scopes_raises(self) -> None:
        with pytest.raises(ValueError, match="scope"):
            _byo(scopes=[])

    def test_non_action_raises(self) -> None:
        with pytest.raises(ValueError, match="action"):
            _byo(action="not-an-action")

    def test_tool_scope_without_tools_raises(self) -> None:
        with pytest.raises(ValueError, match="Tool scope"):
            _byo(scopes=[GuardrailScope.TOOL])

    def test_non_bool_enabled_for_evals_raises(self) -> None:
        with pytest.raises(ValueError, match="enabled_for_evals"):
            _byo(enabled_for_evals="yes")


class TestByoGuardrailHookWiring:
    """Scopes x stage produce the same hooks as the other built-in middlewares."""

    def test_agent_pre_registers_only_before_agent(self) -> None:
        names = _hook_names(
            _byo(name="BYO Guardrail", stage=GuardrailExecutionStage.PRE)
        )
        assert names == ["BYO_Guardrail_before_agent"]

    def test_agent_post_registers_only_after_agent(self) -> None:
        names = _hook_names(
            _byo(name="BYO Guardrail", stage=GuardrailExecutionStage.POST)
        )
        assert names == ["BYO_Guardrail_after_agent"]

    def test_agent_pre_and_post_registers_both(self) -> None:
        names = _hook_names(_byo(name="BYO Guardrail"))
        assert sorted(names) == [
            "BYO_Guardrail_after_agent",
            "BYO_Guardrail_before_agent",
        ]

    def test_llm_pre_registers_only_before_model(self) -> None:
        names = _hook_names(
            _byo(
                name="BYO Guardrail",
                scopes=[GuardrailScope.LLM],
                stage=GuardrailExecutionStage.PRE,
            )
        )
        assert names == ["BYO_Guardrail_before_model"]

    def test_llm_post_registers_only_after_model(self) -> None:
        names = _hook_names(
            _byo(
                name="BYO Guardrail",
                scopes=[GuardrailScope.LLM],
                stage=GuardrailExecutionStage.POST,
            )
        )
        assert names == ["BYO_Guardrail_after_model"]

    def test_tool_scope_registers_one_wrap_tool_call_hook(self) -> None:
        names = _hook_names(_byo(scopes=[GuardrailScope.TOOL], tools=["my_tool"]))
        assert len(names) == 1, f"Expected 1 hook, got: {names}"
        assert "wrap_tool_call" in names[0]

    def test_all_scopes_register_five_hooks(self) -> None:
        names = _hook_names(
            _byo(
                scopes=[
                    GuardrailScope.AGENT,
                    GuardrailScope.LLM,
                    GuardrailScope.TOOL,
                ],
                tools=["my_tool"],
            )
        )
        assert len(names) == 5
        assert sum(1 for n in names if "before" in n) == 2
        assert sum(1 for n in names if "after" in n) == 2
        assert sum(1 for n in names if "wrap_tool_call" in n) == 1


class TestByoGuardrailEvaluationPath:
    """The evaluation path forwards the BYO guardrail to the service."""

    def _passed(self) -> GuardrailValidationResult:
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED, reason=""
        )

    def _failed(self) -> GuardrailValidationResult:
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="Harmful content detected",
        )

    def test_evaluate_forwards_byo_guardrail_to_service(self) -> None:
        middleware = _byo(connection_id=_CONNECTION_ID)
        mock_uipath = MagicMock()
        mock_uipath.guardrails.evaluate_guardrail.return_value = self._passed()
        middleware._uipath = mock_uipath

        result = middleware._evaluate_guardrail("some input")

        assert result.result == GuardrailValidationResultType.PASSED
        mock_uipath.guardrails.evaluate_guardrail.assert_called_once()
        input_data, guardrail = mock_uipath.guardrails.evaluate_guardrail.call_args[0]
        assert input_data == "some input"
        assert guardrail.validator_type == BYO_VALIDATOR_TYPE
        assert guardrail.byo_validator_name == _VALIDATOR_NAME
        assert guardrail.byo_connection_id == _CONNECTION_ID

    def test_block_action_raises_on_validation_failed(self) -> None:
        middleware = _byo(action=_BLOCK)
        with pytest.raises(GuardrailBlockException):
            middleware._handle_validation_result(self._failed(), "bad input")

    def test_log_action_returns_none_on_validation_failed(self) -> None:
        middleware = _byo(action=_LOG)
        assert middleware._handle_validation_result(self._failed(), "bad input") is None

    def test_no_action_on_passed(self) -> None:
        middleware = _byo(action=_BLOCK)
        assert middleware._handle_validation_result(self._passed(), "input") is None
