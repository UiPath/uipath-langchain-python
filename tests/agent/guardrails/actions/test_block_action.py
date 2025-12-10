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
