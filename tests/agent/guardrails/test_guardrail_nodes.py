"""Tests for guardrail node creation and routing."""

import json
import types
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from uipath_langchain.agent.guardrails.guardrail_nodes import (
    create_agent_init_guardrail_node,
    create_agent_terminate_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from uipath_langchain.agent.guardrails.types import (
    ExecutionStage,
)
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState


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


def _patch_uipath(monkeypatch, *, validation_passed=True, reason=None):
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


class TestAgentInitGuardrailNodes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "agent_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "agent_post_execution_example"),
        ],
        ids=["pre-success", "post-success"],
    )
    async def test_agent_init_success_pre_and_post(
        self,
        monkeypatch: MonkeyPatch,
        execution_stage: ExecutionStage,
        expected_name: str,
    ) -> None:
        """Agent init node: routes to success and passes message payload to evaluator."""
        guardrail = MagicMock()
        guardrail.name = "Example"
        fake = _patch_uipath(monkeypatch, validation_passed=True, reason=None)

        node_name, node = create_agent_init_guardrail_node(
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
        assert fake.guardrails.last_text == "payload"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "agent_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "agent_post_execution_example"),
        ],
        ids=["pre-fail", "post-fail"],
    )
    async def test_agent_init_failure_pre_and_post(
        self,
        monkeypatch: MonkeyPatch,
        execution_stage: ExecutionStage,
        expected_name: str,
    ) -> None:
        """Agent init node: routes to failure and sets guardrail_validation_result."""
        guardrail = MagicMock()
        guardrail.name = "Example"
        _patch_uipath(monkeypatch, validation_passed=False, reason="policy_violation")

        node_name, node = create_agent_init_guardrail_node(
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


class TestAgentTerminateGuardrailNodes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "agent_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "agent_post_execution_example"),
        ],
        ids=["pre-success", "post-success"],
    )
    async def test_agent_terminate_success_pre_and_post(
        self,
        monkeypatch: MonkeyPatch,
        execution_stage: ExecutionStage,
        expected_name: str,
    ) -> None:
        """Agent terminate node: routes to success and passes agent_result payload to evaluator."""
        guardrail = MagicMock()
        guardrail.name = "Example"
        fake = _patch_uipath(monkeypatch, validation_passed=True, reason=None)

        node_name, node = create_agent_terminate_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
        )
        assert node_name == expected_name

        agent_result = {"ok": True}
        state = AgentGuardrailsGraphState(messages=[], agent_result=agent_result)
        cmd = await node(state)
        assert cmd.goto == "ok"
        assert cmd.update == {"guardrail_validation_result": None}
        assert fake.guardrails.last_text == str(agent_result)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "agent_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "agent_post_execution_example"),
        ],
        ids=["pre-fail", "post-fail"],
    )
    async def test_agent_terminate_failure_pre_and_post(
        self,
        monkeypatch: MonkeyPatch,
        execution_stage: ExecutionStage,
        expected_name: str,
    ) -> None:
        """Agent terminate node: routes to failure and sets guardrail_validation_result."""
        guardrail = MagicMock()
        guardrail.name = "Example"
        _patch_uipath(monkeypatch, validation_passed=False, reason="policy_violation")

        node_name, node = create_agent_terminate_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
        )
        assert node_name == expected_name

        state = AgentGuardrailsGraphState(messages=[], agent_result={"ok": False})
        cmd = await node(state)
        assert cmd.goto == "nope"
        assert cmd.update == {"guardrail_validation_result": "policy_violation"}


class TestToolGuardrailNodes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "tool_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "tool_post_execution_example"),
        ],
        ids=["pre-success", "post-success"],
    )
    async def test_tool_success_pre_and_post(
        self,
        monkeypatch: MonkeyPatch,
        execution_stage: ExecutionStage,
        expected_name: str,
    ) -> None:
        """Tool node: routes to success and passes the expected payload to evaluator."""
        guardrail = MagicMock()
        guardrail.name = "Example"
        fake = _patch_uipath(monkeypatch, validation_passed=True, reason=None)

        node_name, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        assert node_name == expected_name

        if execution_stage == ExecutionStage.PRE_EXECUTION:
            state = AgentGuardrailsGraphState(
                messages=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "my_tool", "args": {"x": 1}, "id": "call_1"}
                        ],
                    )
                ]
            )
            cmd = await node(state)
            assert cmd.goto == "ok"
            assert cmd.update == {"guardrail_validation_result": None}
            assert json.loads(fake.guardrails.last_text or "{}") == {"x": 1}
        else:
            state = AgentGuardrailsGraphState(
                messages=[ToolMessage(content="tool output", tool_call_id="call_1")]
            )
            cmd = await node(state)
            assert cmd.goto == "ok"
            assert cmd.update == {"guardrail_validation_result": None}
            assert fake.guardrails.last_text == "tool output"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_stage,expected_name",
        [
            (ExecutionStage.PRE_EXECUTION, "tool_pre_execution_example"),
            (ExecutionStage.POST_EXECUTION, "tool_post_execution_example"),
        ],
        ids=["pre-fail", "post-fail"],
    )
    async def test_tool_failure_pre_and_post(
        self,
        monkeypatch: MonkeyPatch,
        execution_stage: ExecutionStage,
        expected_name: str,
    ) -> None:
        """Tool node: routes to failure and sets guardrail_validation_result."""
        guardrail = MagicMock()
        guardrail.name = "Example"
        _patch_uipath(monkeypatch, validation_passed=False, reason="policy_violation")

        node_name, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        assert node_name == expected_name

        if execution_stage == ExecutionStage.PRE_EXECUTION:
            state = AgentGuardrailsGraphState(
                messages=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "my_tool", "args": {"x": 1}, "id": "call_1"}
                        ],
                    )
                ]
            )
        else:
            state = AgentGuardrailsGraphState(
                messages=[ToolMessage(content="tool output", tool_call_id="call_1")]
            )

        cmd = await node(state)
        assert cmd.goto == "nope"
        assert cmd.update == {"guardrail_validation_result": "policy_violation"}
