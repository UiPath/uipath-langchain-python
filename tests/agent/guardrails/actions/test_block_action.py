"""Tests for BlockAction guardrail failure behavior."""

from unittest.mock import MagicMock

import pytest
from uipath.platform.guardrails import GuardrailScope

from uipath_langchain.agent.exceptions import AgentTerminationException
from uipath_langchain.agent.guardrails.actions.block_action import BlockAction
from uipath_langchain.agent.guardrails.types import (
    AgentGuardrailsGraphState,
    ExecutionStage,
)


class TestBlockAction:
    @pytest.mark.asyncio
    async def test_node_name_and_exception_pre_llm(self):
        """PreExecution + LLM: name is sanitized and node raises correct exception."""
        action = BlockAction(reason="Sensitive data detected")
        guardrail = MagicMock()
        guardrail.name = "My Guardrail v1"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="guarded_node_name",
        )

        assert node_name == "llm_pre_execution_my_guardrail_v1_block"

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(AgentGuardrailsGraphState(messages=[]))

        # The exception string is the provided reason
        assert str(excinfo.value) == "Sensitive data detected"

    @pytest.mark.asyncio
    async def test_node_name_and_exception_post_llm(self):
        """PostExecution + LLM: name is sanitized and node raises correct exception."""
        action = BlockAction(reason="Invalid output detected")
        guardrail = MagicMock()
        guardrail.name = "Output Guardrail v2"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="guarded_node_name",
        )

        assert node_name == "llm_post_execution_output_guardrail_v2_block"

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(AgentGuardrailsGraphState(messages=[]))

        # The exception string is the provided reason
        assert str(excinfo.value) == "Invalid output detected"

    @pytest.mark.asyncio
    async def test_node_name_and_exception_pre_tool(self):
        """PreExecution + TOOL: name is sanitized and node raises correct exception."""
        action = BlockAction(reason="Tool input validation failed")
        guardrail = MagicMock()
        guardrail.name = "Tool-Safety@Check"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        assert node_name == "tool_pre_execution_tool_safety_check_block"

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(AgentGuardrailsGraphState(messages=[]))

        # The exception string is the provided reason
        assert str(excinfo.value) == "Tool input validation failed"

    @pytest.mark.asyncio
    async def test_node_name_and_exception_post_tool(self):
        """PostExecution + TOOL: name is sanitized and node raises correct exception."""
        action = BlockAction(reason="Tool output validation failed")
        guardrail = MagicMock()
        guardrail.name = "Output-Validator#2024"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_tool",
        )

        assert node_name == "tool_post_execution_output_validator_2024_block"

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(AgentGuardrailsGraphState(messages=[]))

        # The exception string is the provided reason
        assert str(excinfo.value) == "Tool output validation failed"
