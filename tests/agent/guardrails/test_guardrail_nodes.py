"""Tests for guardrail node creation and routing."""

import json
import types
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from uipath.platform.guardrails import BuiltInValidatorGuardrail

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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
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


class TestGuardrailHelperFunctions:
    """Tests for the refactored helper functions."""

    @pytest.mark.asyncio
    async def test_evaluate_deterministic_guardrail_pre_execution(self, monkeypatch):
        """Test deterministic guardrail evaluation for PRE_EXECUTION."""
        from uipath.core.guardrails import DeterministicGuardrail

        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _evaluate_deterministic_guardrail,
        )

        # Mock the service
        mock_result = types.SimpleNamespace(validation_passed=True, reason=None)
        mock_service = MagicMock()
        mock_service.evaluate_pre_deterministic_guardrail.return_value = mock_result

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.DeterministicGuardrailsService",
            lambda: mock_service,
        )

        guardrail = MagicMock(spec=DeterministicGuardrail)
        state = AgentGuardrailsGraphState(messages=[])
        input_extractor = MagicMock(return_value={"test": "data"})
        output_extractor = MagicMock()

        result = _evaluate_deterministic_guardrail(
            state,
            guardrail,
            ExecutionStage.PRE_EXECUTION,
            input_extractor,
            output_extractor,
        )

        assert result.validation_passed is True
        mock_service.evaluate_pre_deterministic_guardrail.assert_called_once_with(
            input_data={"test": "data"}, guardrail=guardrail
        )

    @pytest.mark.asyncio
    async def test_evaluate_deterministic_guardrail_post_execution(self, monkeypatch):
        """Test deterministic guardrail evaluation for POST_EXECUTION."""
        from uipath.core.guardrails import DeterministicGuardrail

        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _evaluate_deterministic_guardrail,
        )

        # Mock the service
        mock_result = types.SimpleNamespace(validation_passed=False, reason="violation")
        mock_service = MagicMock()
        mock_service.evaluate_post_deterministic_guardrail.return_value = mock_result

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.DeterministicGuardrailsService",
            lambda: mock_service,
        )

        guardrail = MagicMock(spec=DeterministicGuardrail)
        state = AgentGuardrailsGraphState(messages=[])
        input_extractor = MagicMock(return_value={"input": "data"})
        output_extractor = MagicMock(return_value={"output": "data"})

        result = _evaluate_deterministic_guardrail(
            state,
            guardrail,
            ExecutionStage.POST_EXECUTION,
            input_extractor,
            output_extractor,
        )

        assert result.validation_passed is False
        assert result.reason == "violation"
        mock_service.evaluate_post_deterministic_guardrail.assert_called_once_with(
            input_data={"input": "data"},
            output_data={"output": "data"},
            guardrail=guardrail,
        )

    @pytest.mark.asyncio
    async def test_evaluate_builtin_guardrail(self, monkeypatch):
        """Test built-in guardrail evaluation."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _evaluate_builtin_guardrail,
        )

        fake = _patch_uipath(monkeypatch, validation_passed=True, reason=None)

        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        state = AgentGuardrailsGraphState(messages=[HumanMessage("test message")])

        def payload_generator(s):
            return "generated payload"

        result = _evaluate_builtin_guardrail(state, guardrail, payload_generator)

        assert result.validation_passed is True
        assert fake.guardrails.last_text == "generated payload"
        assert fake.guardrails.last_guardrail is guardrail

    def test_create_validation_command_success(self):
        """Test validation command creation for successful validation."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = types.SimpleNamespace(validation_passed=True, reason=None)
        command = _create_validation_command(result, "success_node", "failure_node")

        assert command.goto == "success_node"
        assert command.update == {"guardrail_validation_result": None}

    def test_create_validation_command_failure(self):
        """Test validation command creation for failed validation."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = types.SimpleNamespace(
            validation_passed=False, reason="policy_violation"
        )
        command = _create_validation_command(result, "success_node", "failure_node")

        assert command.goto == "failure_node"
        assert command.update == {"guardrail_validation_result": "policy_violation"}

    @pytest.mark.asyncio
    async def test_unsupported_guardrail_type_raises_error(self):
        """Test that unsupported guardrail types raise an error."""
        from uipath_langchain.agent.exceptions import AgentTerminationException
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            create_llm_guardrail_node,
        )

        # Create a mock that doesn't match any supported type
        guardrail = MagicMock()  # No spec, so isinstance checks will fail
        guardrail.name = "UnsupportedGuardrail"

        node_name, node = create_llm_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        state = AgentGuardrailsGraphState(messages=[HumanMessage("test")])

        with pytest.raises(AgentTerminationException) as exc_info:
            await node(state)

        error_message = str(exc_info.value)
        assert "is not supported" in error_message
        assert "MagicMock" in error_message
        assert "DeterministicGuardrail" in error_message
        assert "BuiltInValidatorGuardrail" in error_message
