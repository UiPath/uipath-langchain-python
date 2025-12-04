"""Tests for BlockAction guardrail failure behavior."""

import types

import pytest
from uipath.platform.guardrails import GuardrailScope

from uipath_langchain.agent.exceptions import AgentTerminationException
from uipath_langchain.agent.guardrails.actions.block_action import BlockAction


class TestBlockAction:
    @pytest.mark.asyncio
    async def test_node_name_and_exception_pre_llm(self):
        """PreExecution + LLM: name is sanitized and node raises correct exception."""
        action = BlockAction(reason="Sensitive data detected")
        guardrail = types.SimpleNamespace(name="My Guardrail (v1)")

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage="PreExecution",
        )

        assert node_name == "My_Guardrail_v1_preexecution_llm_block"

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(types.SimpleNamespace())

        # The exception string is the provided reason
        assert str(excinfo.value) == "Sensitive data detected"

