"""Tests for LogAction guardrail failure behavior."""

import logging
from itertools import product
from unittest.mock import MagicMock

import pytest
from uipath.platform.guardrails import GuardrailScope

from uipath_langchain.agent.guardrails.actions.log_action import LogAction
from uipath_langchain.agent.guardrails.types import (
    ExecutionStage,
)
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState


class TestLogAction:
    @pytest.mark.asyncio
    async def test_node_name_and_logs_custom_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PreExecution + LLM: name is sanitized and custom message is logged at given level."""
        action = LogAction(message="custom message", level=logging.ERROR)
        guardrail = MagicMock()
        guardrail.name = "My Guardrail v1"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="guarded_node_name",
        )

        assert node_name == "llm_pre_execution_my_guardrail_v1_log"

        with caplog.at_level(logging.ERROR):
            result = await node(
                AgentGuardrailsGraphState(
                    messages=[], guardrail_validation_result="ignored"
                )
            )

        assert result == {}
        # Verify the exact custom message was logged at ERROR
        assert any(
            rec.levelno == logging.ERROR and rec.message == "custom message"
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage", "level"),
        list(
            product(
                (GuardrailScope.LLM, GuardrailScope.AGENT, GuardrailScope.TOOL),
                (ExecutionStage.PRE_EXECUTION, ExecutionStage.POST_EXECUTION),
                (logging.INFO, logging.WARNING, logging.ERROR),
            )
        ),
    )
    async def test_default_message_includes_context(
        self,
        caplog: pytest.LogCaptureFixture,
        scope: GuardrailScope,
        stage: ExecutionStage,
        level: int,
    ) -> None:
        """Default message includes guardrail name, scope, stage, and reason."""
        action = LogAction(message=None, level=level)
        guardrail = MagicMock()
        guardrail.name = "My Guardrail"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=stage,
            guarded_component_name="guarded_node_name",
        )
        assert (
            node_name == f"{scope.name.lower()}_{stage.name.lower()}_my_guardrail_log"
        )

        with caplog.at_level(level):
            result = await node(
                AgentGuardrailsGraphState(
                    messages=[], guardrail_validation_result="bad input"
                )
            )

        assert result == {}
        # Confirm default formatted message content
        expected = (
            "Guardrail [My Guardrail] validation failed for "
            f"[{scope.name}] [{stage.name}] with the following reason: bad input"
        )
        assert any(
            rec.levelno == level and rec.message == expected for rec in caplog.records
        )
