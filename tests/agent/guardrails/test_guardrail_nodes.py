"""Tests for guardrail node creation and routing."""

import types
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from uipath_langchain.agent.guardrails.guardrail_nodes import (
    create_llm_guardrail_node,
)
from uipath_langchain.agent.guardrails.types import (
    AgentGuardrailsGraphState,
    ExecutionStage,
)


class FakeGuardrails:
    def __init__(self, result):
        self._result = result
        self.last_text = None
        self.last_guardrail = None

    def evaluate_guardrail(self, text, guardrail):
        self.last_text = text
        self.last_guardrail = guardrail
        return self._result


class FakeUiPath:
    def __init__(self, result):
        self.guardrails = FakeGuardrails(result)


def _patch_uipath(monkeypatch, validation_passed=True, reason=None):
    result = types.SimpleNamespace(validation_passed=validation_passed, reason=reason)
    fake = FakeUiPath(result)
    monkeypatch.setattr(
        "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath",
        lambda: fake,
    )
    return fake


class TestLlmGuardrailNodes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "llm_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "llm_post_execution_example"),
        ],
        ids=["pre-success", "post-success"],
    )
    async def test_llm_success_pre_and_post(
        self,
        monkeypatch,
        execution_stage: ExecutionStage,
        expected_name,
    ):
        guardrail = MagicMock()
        guardrail.name = "Example"
        _patch_uipath(monkeypatch, validation_passed=True, reason=None)
        node_name, node = create_llm_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
        )
        assert node_name == expected_name
        state = AgentGuardrailsGraphState(messages=[HumanMessage("payload")])
        cmd = await node(state)
        assert cmd.goto == "ok"
        assert cmd.update == {"guardrail_validation_result": None}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "llm_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "llm_post_execution_example"),
        ],
        ids=["pre-fail", "post-fail"],
    )
    async def test_llm_failure_pre_and_post(
        self,
        monkeypatch,
        execution_stage: ExecutionStage,
        expected_name,
    ):
        guardrail = MagicMock()
        guardrail.name = "Example"
        _patch_uipath(monkeypatch, validation_passed=False, reason="policy_violation")
        node_name, node = create_llm_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
        )
        assert node_name == expected_name
        state = AgentGuardrailsGraphState(messages=[SystemMessage("payload")])
        cmd = await node(state)
        assert cmd.goto == "nope"
        assert cmd.update == {"guardrail_validation_result": "policy_violation"}
