"""Tests for Bring Your Own Guardrail (BYOG) decorator support.

Two groups:

1. **Adapter evaluator-error logging** — the LLM/agent wrappers swallow
   evaluator exceptions (fail-open); they must log a WARNING naming the
   guardrail so a misconfigured BYOG configuration (``PROVIDER_ERROR``,
   deleted/disabled config, BYOG not enabled) is visible instead of silent.

2. **``ByoValidator`` through ``@guardrail``** — the validator reaches the
   guardrails service as a ``byo`` guardrail carrying ``byoValidatorName`` /
   ``byoConnectionId``, for a tool, a plain function, and a validator reused
   across targets.
"""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)

from uipath_langchain.guardrails import ByoValidator, LogAction, guardrail
from uipath_langchain.guardrails._langchain_adapter import (
    _apply_agent_input_guardrail,
    _apply_agent_output_guardrail,
    _apply_llm_post,
    _apply_llm_pre,
)

_ADAPTER_LOGGER = "uipath_langchain.guardrails._langchain_adapter"

_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)


def _raise_eval(*_args: Any, **_kwargs: Any) -> GuardrailValidationResult:
    raise RuntimeError("evaluator boom")


class TestByoValidatorExport:
    def test_exported_from_guardrails_package(self) -> None:
        from uipath_langchain import guardrails

        assert "ByoValidator" in guardrails.__all__

    def test_exported_from_decorators_shim(self) -> None:
        from uipath_langchain.guardrails import decorators

        assert "ByoValidator" in decorators.__all__


class TestAdapterEvaluatorErrorLogging:
    """Evaluator failures stay fail-open but are logged with the guardrail name."""

    def test_llm_pre_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER):
            _apply_llm_pre(
                [HumanMessage(content="hi")], _raise_eval, LogAction(), "my-byog"
            )
        assert any("my-byog" in r.message for r in caplog.records)

    def test_llm_post_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER):
            _apply_llm_post(
                AIMessage(content="ans"), _raise_eval, LogAction(), "my-byog"
            )
        assert any("my-byog" in r.message for r in caplog.records)

    def test_agent_input_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER):
            _apply_agent_input_guardrail(
                {"messages": [HumanMessage(content="hi")]},
                _raise_eval,
                LogAction(),
                "my-byog",
            )
        assert any("my-byog" in r.message for r in caplog.records)

    def test_agent_output_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER):
            _apply_agent_output_guardrail(
                {"messages": [AIMessage(content="ans")]},
                _raise_eval,
                LogAction(),
                "my-byog",
            )
        assert any("my-byog" in r.message for r in caplog.records)

    def test_exception_traceback_is_included(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER):
            _apply_llm_pre([HumanMessage(content="hi")], _raise_eval, LogAction(), "g")
        assert any(r.exc_info for r in caplog.records)


class TestByoValidatorThroughDecorator:
    """ByoValidator + @guardrail on LangChain targets (adapter path)."""

    def _mock_uipath(self) -> MagicMock:
        mock_uipath = MagicMock()
        mock_uipath.guardrails.evaluate_guardrail.return_value = _PASSED
        return mock_uipath

    def test_tool_scope_forwards_byo_guardrail(self) -> None:
        @guardrail(
            validator=ByoValidator(
                "my-harmful-content-guardrail",
                connection_id="my-byog-guardrail-connection",
            ),
            action=LogAction(),
            name="BYOG tool guardrail",
        )
        @tool
        def echo(text: str) -> str:
            """Echo the input text back."""
            return f"echo: {text}"

        mock_uipath = self._mock_uipath()
        with patch("uipath.platform.UiPath", return_value=mock_uipath):
            result = echo.invoke({"text": "hello"})

        assert result == "echo: hello"
        assert mock_uipath.guardrails.evaluate_guardrail.called
        _, g = mock_uipath.guardrails.evaluate_guardrail.call_args[0]
        assert g.validator_type == "byo"
        assert g.byo_validator_name == "my-harmful-content-guardrail"
        assert g.byo_connection_id == "my-byog-guardrail-connection"
        assert g.name == "BYOG tool guardrail"

    def test_plain_function_forwards_byo_guardrail(self) -> None:
        @guardrail(
            validator=ByoValidator("byog-pii"),
            action=LogAction(),
        )
        def summarize(text: str) -> str:
            return text.upper()

        mock_uipath = self._mock_uipath()
        with patch("uipath.platform.UiPath", return_value=mock_uipath):
            assert summarize("hello") == "HELLO"

        assert mock_uipath.guardrails.evaluate_guardrail.called
        _, g = mock_uipath.guardrails.evaluate_guardrail.call_args[0]
        assert g.validator_type == "byo"
        assert g.byo_validator_name == "byog-pii"
        assert g.byo_connection_id is None

    def test_validator_is_reusable_across_targets(self) -> None:
        shared = ByoValidator("byog-shared", connection_id="conn-1")

        @guardrail(validator=shared, action=LogAction())
        def first(text: str) -> str:
            return text

        @guardrail(validator=shared, action=LogAction())
        def second(text: str) -> str:
            return text

        mock_uipath = self._mock_uipath()
        with patch("uipath.platform.UiPath", return_value=mock_uipath):
            first("a")
            second("b")

        calls = mock_uipath.guardrails.evaluate_guardrail.call_args_list
        assert len(calls) >= 2
        for call in calls:
            _, g = call[0]
            assert g.byo_validator_name == "byog-shared"
            assert g.byo_connection_id == "conn-1"
