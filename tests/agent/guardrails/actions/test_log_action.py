"""Tests for LogAction guardrail failure behavior."""

import logging
from unittest.mock import MagicMock

import pytest
from uipath.platform.guardrails import GuardrailScope

from uipath_langchain.agent.guardrails.actions.log_action import LogAction
from uipath_langchain.agent.guardrails.types import (
    AgentGuardrailsGraphState,
    ExecutionStage,
)


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
    async def test_default_message_includes_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PostExecution + TOOL: default message includes guardrail name, scope, stage, and reason."""
        action = LogAction(message=None, level=logging.WARNING)
        guardrail = MagicMock()
        guardrail.name = "My Guardrail"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="guarded_node_name",
        )
        assert node_name == "tool_post_execution_my_guardrail_log"

        with caplog.at_level(logging.WARNING):
            result = await node(
                AgentGuardrailsGraphState(
                    messages=[], guardrail_validation_result="bad input"
                )
            )

        assert result == {}
        # Confirm default formatted message content
        assert any(
            rec.levelno == logging.WARNING
            and rec.message
            == "Guardrail [My Guardrail] validation failed for [TOOL] [POST_EXECUTION] with the following reason: bad input"
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_node_name_and_exception_post_llm(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PostExecution + LLM: name is sanitized and default message is logged."""
        action = LogAction(message=None, level=logging.INFO)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="guarded_node_name",
        )

        # Verify node name format
        assert node_name == "llm_post_execution_test_guardrail_log"
        assert isinstance(node_name, str)
        assert node_name.endswith("_log")
        assert "llm" in node_name
        assert "post_execution" in node_name

        # Verify node is callable
        assert callable(node)

        # Verify node returns empty dict
        with caplog.at_level(logging.INFO):
            await node(
                AgentGuardrailsGraphState(
                    messages=[], guardrail_validation_result="validation error"
                )
            )

        # Verify log record properties
        log_record = caplog.records[0]
        assert log_record.levelno == logging.INFO

        # Verify default message includes all context
        assert (
            "Guardrail [Test Guardrail] validation failed for [LLM] [POST_EXECUTION]"
            in log_record.message
        )
        assert "validation error" in log_record.message
        assert (
            log_record.message
            == "Guardrail [Test Guardrail] validation failed for [LLM] [POST_EXECUTION] with the following reason: validation error"
        )

    @pytest.mark.asyncio
    async def test_node_name_and_exception_pre_tool(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PreExecution + TOOL: name is sanitized and default message is logged."""
        action = LogAction(message=None, level=logging.WARNING)
        guardrail = MagicMock()
        guardrail.name = "Tool Guardrail v2"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        # Verify node name format
        assert node_name == "tool_pre_execution_tool_guardrail_v2_log"
        assert isinstance(node_name, str)
        assert node_name.endswith("_log")
        assert "tool" in node_name
        assert "pre_execution" in node_name

        # Verify node returns empty dict
        with caplog.at_level(logging.WARNING):
            await node(
                AgentGuardrailsGraphState(
                    messages=[], guardrail_validation_result="invalid tool args"
                )
            )

        # Verify log record properties
        log_record = caplog.records[0]
        assert log_record.levelno == logging.WARNING

        # Verify default message includes all context
        assert (
            "Guardrail [Tool Guardrail v2] validation failed for [TOOL] [PRE_EXECUTION]"
            in log_record.message
        )
        assert "invalid tool args" in log_record.message
        assert (
            log_record.message
            == "Guardrail [Tool Guardrail v2] validation failed for [TOOL] [PRE_EXECUTION] with the following reason: invalid tool args"
        )

    @pytest.mark.asyncio
    async def test_node_name_and_exception_post_tool(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PostExecution + TOOL: name is sanitized and custom message is logged."""
        action = LogAction(message="Tool execution failed", level=logging.ERROR)
        guardrail = MagicMock()
        guardrail.name = "Special-Tool@Guardrail"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_tool",
        )

        # Verify node name format (special characters are sanitized)
        assert node_name == "tool_post_execution_special_tool_guardrail_log"
        assert isinstance(node_name, str)
        assert node_name.endswith("_log")
        assert "tool" in node_name
        assert "post_execution" in node_name

        # Verify node returns empty dict
        with caplog.at_level(logging.ERROR):
            await node(
                AgentGuardrailsGraphState(
                    messages=[], guardrail_validation_result="tool error"
                )
            )

        # Verify log record properties
        log_record = caplog.records[0]
        assert log_record.levelno == logging.ERROR

        # Verify custom message was logged (not default message)
        assert log_record.message == "Tool execution failed"
        assert (
            "Guardrail" not in log_record.message
        )  # Custom message doesn't include guardrail context
        assert (
            "validation failed" not in log_record.message
        )  # Custom message doesn't include default format

    @pytest.mark.asyncio
    async def test_node_name_and_exception_post_tool_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PostExecution + TOOL: name is sanitized and default message is logged at WARNING level."""
        action = LogAction(message=None, level=logging.WARNING)
        guardrail = MagicMock()
        guardrail.name = "Post Tool Guardrail"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_tool",
        )

        # Verify node name format
        assert node_name == "tool_post_execution_post_tool_guardrail_log"
        assert isinstance(node_name, str)
        assert node_name.endswith("_log")
        assert "tool" in node_name
        assert "post_execution" in node_name

        # Verify node returns empty dict
        with caplog.at_level(logging.WARNING):
            await node(
                AgentGuardrailsGraphState(
                    messages=[], guardrail_validation_result="post execution error"
                )
            )

        # Verify log record properties
        log_record = caplog.records[0]
        assert log_record.levelno == logging.WARNING

        # Verify default message includes all context
        assert (
            "Guardrail [Post Tool Guardrail] validation failed for [TOOL] [POST_EXECUTION]"
            in log_record.message
        )
        assert "post execution error" in log_record.message
        assert (
            log_record.message
            == "Guardrail [Post Tool Guardrail] validation failed for [TOOL] [POST_EXECUTION] with the following reason: post execution error"
        )
