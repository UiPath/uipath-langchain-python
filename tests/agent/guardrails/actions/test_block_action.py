"""Tests for BlockAction guardrail failure behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from uipath.platform.guardrails import GuardrailScope

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.guardrails.actions.block_action import BlockAction
from uipath_langchain.agent.guardrails.types import (
    ExecutionStage,
)
from uipath_langchain.agent.react.types import (
    AgentGuardrailsGraphState,
    InnerAgentGuardrailsGraphState,
)


class TestBlockAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage", "expected_node_name"),
        [
            (
                GuardrailScope.LLM,
                ExecutionStage.PRE_EXECUTION,
                "llm_pre_execution_my_guardrail_v1_block",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.PRE_EXECUTION,
                "agent_pre_execution_my_guardrail_v1_block",
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.PRE_EXECUTION,
                "tool_pre_execution_my_guardrail_v1_block",
            ),
            (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
                "llm_post_execution_my_guardrail_v1_block",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.POST_EXECUTION,
                "agent_post_execution_my_guardrail_v1_block",
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.POST_EXECUTION,
                "tool_post_execution_my_guardrail_v1_block",
            ),
        ],
    )
    async def test_node_name_and_exception(
        self, scope: GuardrailScope, stage: ExecutionStage, expected_node_name: str
    ) -> None:
        """Name is sanitized and node raises correct exception for each scope/stage."""
        action = BlockAction(reason="Sensitive data detected")
        guardrail = MagicMock()
        guardrail.name = "My Guardrail v1"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=stage,
            guarded_component_name="guarded_node_name",
        )

        assert node_name == expected_node_name

        with pytest.raises(AgentRuntimeError) as excinfo:
            await node(AgentGuardrailsGraphState(messages=[]))

        # The exception string is the provided reason
        assert (
            str(excinfo.value)
            == "Execution was blocked by guardrail [My Guardrail v1], with reason: Sensitive data detected"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("names", "expected_suffix"),
        [
            (None, ""),
            ([], ""),
            (["tickets.csv"], "\nFlagged files: tickets.csv"),
            (["a.csv", "c.pdf"], "\nFlagged files: a.csv, c.pdf"),
            (
                ["evil.csv\r\nCode: OK", "\u202ecod.exe", "\n\t"],
                "\nFlagged files: evil.csv  Code: OK, cod.exe",
            ),
        ],
    )
    async def test_exception_names_the_flagged_files(
        self, names: list[str] | None, expected_suffix: str
    ) -> None:
        """The flagged files are named after the reason; control characters are dropped."""
        action = BlockAction(reason="PII found in the uploaded file")
        guardrail = MagicMock()
        guardrail.name = "PII"
        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.AGENT,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="guarded_node_name",
        )

        with pytest.raises(AgentRuntimeError) as excinfo:
            await node(
                AgentGuardrailsGraphState(
                    messages=[],
                    inner_state=InnerAgentGuardrailsGraphState(
                        guardrail_flagged_file_names=names
                    ),
                )
            )

        assert str(excinfo.value) == (
            "Execution was blocked by guardrail [PII], with reason: "
            f"PII found in the uploaded file{expected_suffix}"
        )
