"""Tests for guardrail node creation and routing."""

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from uipath_langchain.agent.guardrails.guardrail_nodes import (
    _create_guardrail_node,
    create_agent_init_guardrail_node,
    create_agent_terminate_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from uipath_langchain.agent.guardrails.types import (
    ExecutionStage,
)
from uipath_langchain.agent.react.types import (
    AgentGuardrailsGraphState,
    InnerAgentGuardrailsGraphState,
)


class FakeGuardrails:
    def __init__(self, result):
        self._result = result
        self.last_text = None
        self.last_guardrail = None
        self.last_attachments = None
        self.call_count = 0

    def evaluate_guardrail(
        self, text, guardrail, *, attachments=None, termination_mode=None
    ):
        self.call_count += 1
        self.last_text = text
        self.last_guardrail = guardrail
        self.last_attachments = attachments
        self.last_termination_mode = termination_mode
        return self._result


class FakeUiPath:
    def __init__(self, result):
        self.guardrails = FakeGuardrails(result)


def _patch_uipath(
    monkeypatch,
    *,
    result: GuardrailValidationResultType = GuardrailValidationResultType.PASSED,
    reason: str = "",
):
    """Create a fake UiPath instance with a guardrail validation result.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        result: The validation result type.
        reason: The reason for the validation result.

    Returns:
        FakeUiPath instance with the specified validation result.
    """
    validation_result = GuardrailValidationResult(
        result=result,
        reason=reason,
    )
    fake = FakeUiPath(validation_result)
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
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="validation passed",
        )
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
        assert cmd.update == {
            "inner_state": {
                "guardrail_validation_result": True,
                "guardrail_validation_details": "validation passed",
                "guardrail_flagged_file_names": None,
            }
        }

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
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="policy_violation",
        )
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
        assert cmd.update == {
            "inner_state": {
                "guardrail_validation_result": False,
                "guardrail_validation_details": "policy_violation",
                "guardrail_flagged_file_names": None,
            }
        }


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
        fake = _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="validation passed",
        )

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
        assert cmd.update == {
            "inner_state": {
                "guardrail_validation_result": True,
                "guardrail_validation_details": "validation passed",
                "guardrail_flagged_file_names": None,
            }
        }
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
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="policy_violation",
        )

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
        assert cmd.update == {
            "inner_state": {
                "guardrail_validation_result": False,
                "guardrail_validation_details": "policy_violation",
                "guardrail_flagged_file_names": None,
            }
        }


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
        fake = _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="validation passed",
        )

        node_name, node = create_agent_terminate_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
        )
        assert node_name == expected_name

        agent_result = {"ok": True}
        state = AgentGuardrailsGraphState(
            messages=[],
            inner_state=InnerAgentGuardrailsGraphState(agent_result=agent_result),
        )
        cmd = await node(state)
        assert cmd.goto == "ok"
        assert cmd.update == {
            "inner_state": {
                "guardrail_validation_result": True,
                "guardrail_validation_details": "validation passed",
                "guardrail_flagged_file_names": None,
            }
        }
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
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="policy_violation",
        )

        node_name, node = create_agent_terminate_guardrail_node(
            guardrail=guardrail,
            execution_stage=execution_stage,
            success_node="ok",
            failure_node="nope",
        )
        assert node_name == expected_name

        state = AgentGuardrailsGraphState(
            messages=[],
            inner_state=InnerAgentGuardrailsGraphState(agent_result={"ok": False}),
        )
        cmd = await node(state)
        assert cmd.goto == "nope"
        assert cmd.update == {
            "inner_state": {
                "guardrail_validation_result": False,
                "guardrail_validation_details": "policy_violation",
                "guardrail_flagged_file_names": None,
            }
        }


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
        fake = _patch_uipath(
            monkeypatch, result=GuardrailValidationResultType.PASSED, reason=""
        )

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
            assert cmd.update == {
                "inner_state": {
                    "guardrail_validation_result": True,
                    "guardrail_validation_details": "",
                    "guardrail_flagged_file_names": None,
                }
            }
            assert json.loads(fake.guardrails.last_text or "{}") == {"x": 1}
        else:
            state = AgentGuardrailsGraphState(
                messages=[ToolMessage(content="tool output", tool_call_id="call_1")]
            )
            cmd = await node(state)
            assert cmd.goto == "ok"
            assert cmd.update == {
                "inner_state": {
                    "guardrail_validation_result": True,
                    "guardrail_validation_details": "",
                    "guardrail_flagged_file_names": None,
                }
            }
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
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="policy_violation",
        )

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
        assert cmd.update == {
            "inner_state": {
                "guardrail_validation_result": False,
                "guardrail_validation_details": "policy_violation",
                "guardrail_flagged_file_names": None,
            }
        }


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
        mock_result = GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED,
            reason="",
        )
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

        assert result.result == GuardrailValidationResultType.PASSED
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
        mock_result = GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="violation",
        )
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

        assert result.result == GuardrailValidationResultType.VALIDATION_FAILED
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

        fake = _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="validation passed",
        )

        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)

        result = await _evaluate_builtin_guardrail(guardrail, "generated payload")

        assert result.result == GuardrailValidationResultType.PASSED
        assert fake.guardrails.last_text == "generated payload"
        assert fake.guardrails.last_guardrail is guardrail
        assert fake.guardrails.last_attachments is None

    @pytest.mark.asyncio
    async def test_evaluate_builtin_guardrail_forwards_attachments(self, monkeypatch):
        """Attachment references reach the SDK call."""
        from uipath.platform.guardrails import GuardrailAttachment

        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _evaluate_builtin_guardrail,
        )

        fake = _patch_uipath(monkeypatch)
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        attachment = GuardrailAttachment(
            id="7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10",
            file_name="a.csv",
            mime_type="text/csv",
        )

        await _evaluate_builtin_guardrail(guardrail, "payload", [attachment])

        assert fake.guardrails.last_attachments == [attachment]

    def test_create_validation_command_success(self):
        """Test validation command creation for successful validation."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED,
            reason="validation passed",
        )
        command = _create_validation_command(result, "success_node", "failure_node")

        assert command.goto == "success_node"
        assert command.update == {
            "inner_state": {
                "guardrail_validation_result": True,
                "guardrail_validation_details": "validation passed",
                "guardrail_flagged_file_names": None,
            }
        }

    def test_create_validation_command_failure(self):
        """Test validation command creation for failed validation."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="policy_violation",
        )
        command = _create_validation_command(result, "success_node", "failure_node")

        assert command.goto == "failure_node"
        assert command.update == {
            "inner_state": {
                "guardrail_validation_result": False,
                "guardrail_validation_details": "policy_violation",
                "guardrail_flagged_file_names": None,
            }
        }

    def test_create_validation_command_success_with_span_id(self):
        """Test validation command includes guardrail_span_id when span_id is present."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED,
            reason="validation passed",
        )
        result.span_id = "span-123"
        command = _create_validation_command(result, "success_node", "failure_node")

        assert command.goto == "success_node"
        assert command.update == {
            "inner_state": {
                "guardrail_validation_result": True,
                "guardrail_validation_details": "validation passed",
                "guardrail_flagged_file_names": None,
                "guardrail_span_id": "span-123",
            }
        }

    def test_create_validation_command_failure_with_span_id(self):
        """Test validation command includes guardrail_span_id on failure when span_id is present."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="policy_violation",
        )
        result.span_id = "span-456"
        command = _create_validation_command(result, "success_node", "failure_node")

        assert command.goto == "failure_node"
        assert command.update == {
            "inner_state": {
                "guardrail_validation_result": False,
                "guardrail_validation_details": "policy_violation",
                "guardrail_flagged_file_names": None,
                "guardrail_span_id": "span-456",
            }
        }

    def test_create_validation_command_without_span_id(self):
        """Test validation command excludes guardrail_span_id when span_id is absent."""
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED,
            reason="validation passed",
        )
        command = _create_validation_command(result, "success_node", "failure_node")

        assert command.goto == "success_node"
        assert command.update is not None
        assert "guardrail_span_id" not in command.update["inner_state"]

    def test_create_validation_command_feature_disabled_raises_exception(self):
        """Test that FEATURE_DISABLED result raises AgentRuntimeError."""
        from uipath.runtime.errors import UiPathErrorCategory

        from uipath_langchain.agent.exceptions import AgentRuntimeError
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = GuardrailValidationResult(
            result=GuardrailValidationResultType.FEATURE_DISABLED,
            reason="Guardrail feature is disabled",
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            _create_validation_command(result, "success_node", "failure_node")

        assert exc_info.value.error_info.title == "Guardrail validation error"
        assert "Guardrail feature is disabled" in exc_info.value.error_info.detail
        assert exc_info.value.error_info.category == UiPathErrorCategory.DEPLOYMENT

    def test_create_validation_command_entitlements_missing_raises_exception(self):
        """Test that ENTITLEMENTS_MISSING result raises AgentRuntimeError."""
        from uipath.runtime.errors import UiPathErrorCategory

        from uipath_langchain.agent.exceptions import AgentRuntimeError
        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _create_validation_command,
        )

        result = GuardrailValidationResult(
            result=GuardrailValidationResultType.ENTITLEMENTS_MISSING,
            reason="Guardrail entitlement is missing",
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            _create_validation_command(result, "success_node", "failure_node")

        assert exc_info.value.error_info.title == "Guardrail validation error"
        assert "Guardrail entitlement is missing" in exc_info.value.error_info.detail
        assert exc_info.value.error_info.category == UiPathErrorCategory.DEPLOYMENT

    @pytest.mark.asyncio
    async def test_unsupported_guardrail_type_raises_error(self):
        """Test that unsupported guardrail types raise an error."""
        from uipath_langchain.agent.exceptions import AgentRuntimeError
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

        with pytest.raises(AgentRuntimeError) as exc_info:
            await node(state)

        error_message = str(exc_info.value)
        assert "is not supported" in error_message
        assert "MagicMock" in error_message
        assert "DeterministicGuardrail" in error_message
        assert "BuiltInValidatorGuardrail" in error_message


class TestGuardrailNodeMetadata:
    """Tests for guardrail node __metadata__ attribute for observability."""

    def test_llm_guardrail_node_has_metadata(self):
        """Test that LLM guardrail node has __metadata__ attribute."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"
        guardrail.description = "Test description"

        _, node = create_llm_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        assert hasattr(node, "__metadata__")
        assert isinstance(node.__metadata__, dict)

    def test_llm_guardrail_node_metadata_fields(self):
        """Test that LLM guardrail node has correct metadata fields."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"
        guardrail.description = "Test description"

        _, node = create_llm_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["guardrail"] == guardrail
        assert metadata["scope"] == "Llm"
        assert metadata["execution_stage"] == "preExecution"
        assert metadata["tool_name"] is None
        assert metadata["tool_type"] is None

    def test_tool_guardrail_node_has_tool_name(self):
        """Test that TOOL scope guardrail has tool_name in metadata."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"

        _, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["scope"] == "Tool"
        assert metadata["tool_name"] == "my_tool"

    def test_tool_guardrail_node_has_tool_type(self):
        """Test that TOOL scope guardrail has tool_type in metadata."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"

        _, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
            tool_type="process",
        )

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["tool_type"] == "process"

    def test_tool_guardrail_node_tool_type_defaults_to_none(self):
        """Test that tool_type defaults to None when not provided."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"

        _, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["tool_type"] is None

    def test_agent_init_guardrail_node_metadata(self):
        """Test that AGENT init guardrail has correct scope in metadata."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"

        _, node = create_agent_init_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.POST_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["scope"] == "Agent"
        assert metadata["execution_stage"] == "postExecution"

    @pytest.mark.asyncio
    async def test_builtin_guardrail_payload_populated_pre_execution(self, monkeypatch):
        """Test that payload.input is populated for builtin guardrail at PRE_EXECUTION."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="",
        )

        _, node = create_llm_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        state = AgentGuardrailsGraphState(messages=[HumanMessage("test input")])
        await node(state)

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["payload"]["input"] == "test input"
        assert metadata["payload"]["output"] is None

    @pytest.mark.asyncio
    async def test_builtin_guardrail_payload_populated_post_execution(
        self, monkeypatch
    ):
        """Test that payload.output is populated for builtin guardrail at POST_EXECUTION."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="",
        )

        _, node = create_llm_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.POST_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        state = AgentGuardrailsGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "tool", "args": {"x": 1}, "id": "1"}],
                )
            ]
        )
        await node(state)

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["payload"]["output"] is not None
        assert metadata["payload"]["input"] is None

    @pytest.mark.asyncio
    async def test_tool_guardrail_payload_populated_pre_execution(self, monkeypatch):
        """Test that payload.input is populated for tool guardrail at PRE_EXECUTION."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="",
        )

        _, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )

        state = AgentGuardrailsGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "my_tool", "args": {"param": "value"}, "id": "call_1"}
                    ],
                )
            ]
        )
        await node(state)

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["payload"]["input"] == '{"param": "value"}'
        assert metadata["payload"]["output"] is None

    @pytest.mark.asyncio
    async def test_tool_guardrail_payload_populated_post_execution(self, monkeypatch):
        """Test that payload.output is populated for tool guardrail at POST_EXECUTION."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "TestGuardrail"
        _patch_uipath(
            monkeypatch,
            result=GuardrailValidationResultType.PASSED,
            reason="",
        )

        _, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.POST_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )

        state = AgentGuardrailsGraphState(
            messages=[ToolMessage(content="tool output data", tool_call_id="call_1")]
        )
        await node(state)

        metadata = getattr(node, "__metadata__", None)
        assert metadata is not None
        assert metadata["payload"]["output"] == "tool output data"
        assert metadata["payload"]["input"] is None


class TestGuardrailNodeAttachments:
    """Built-in guardrail nodes forward attachment references to the judge: the run's
    registry at Agent and LLM scope, the files one tool call mentions at Tool scope."""

    _UUID = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"

    @staticmethod
    def _judge_guardrail() -> MagicMock:
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "Example"
        guardrail.validator_type = "llm_as_judge"
        return guardrail

    @staticmethod
    def _patch_resolver(monkeypatch, attachments):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.resolve_guardrail_attachments",
            AsyncMock(return_value=attachments),
        )

    def _state_with_attachment(self):
        from uipath.platform.attachments import Attachment

        return AgentGuardrailsGraphState(
            messages=[HumanMessage("payload")],
            inner_state=InnerAgentGuardrailsGraphState(
                job_attachments={
                    self._UUID: Attachment(
                        id=uuid.UUID(self._UUID),
                        full_name="Tickets.csv",
                        mime_type="text/csv",
                    )
                }
            ),
        )

    @pytest.mark.asyncio
    async def test_agent_init_node_forwards_resolved_attachments(self, monkeypatch):
        """An Agent-scope PRE guardrail sees the file supplied as agent input."""
        from uipath.platform.guardrails import GuardrailAttachment

        fake = _patch_uipath(monkeypatch, reason="ok")
        attachment = GuardrailAttachment(
            id=self._UUID,
            file_name="Tickets.csv",
            mime_type="text/csv",
        )
        self._patch_resolver(monkeypatch, [attachment])

        _, node = create_agent_init_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        cmd = await node(self._state_with_attachment())

        assert cmd.goto == "ok"
        assert fake.guardrails.last_attachments == [attachment]

    @pytest.mark.asyncio
    async def test_llm_node_forwards_resolved_attachments(self, monkeypatch):
        from uipath.platform.guardrails import GuardrailAttachment

        fake = _patch_uipath(monkeypatch, reason="ok")
        attachment = GuardrailAttachment(
            id=self._UUID,
            file_name="Tickets.csv",
            mime_type="text/csv",
        )
        self._patch_resolver(monkeypatch, [attachment])

        _, node = create_llm_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        await node(self._state_with_attachment())

        assert fake.guardrails.last_attachments == [attachment]

    def _tool_pre_state(self, args):
        return AgentGuardrailsGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "my_tool", "args": args, "id": "c1"}],
                )
            ],
            inner_state=InnerAgentGuardrailsGraphState(
                job_attachments=self._state_with_attachment().inner_state.job_attachments
            ),
        )

    def _tool_post_state(self, content):
        return AgentGuardrailsGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "my_tool", "args": {}, "id": "c1"}],
                ),
                ToolMessage(content=content, tool_call_id="c1"),
            ],
            inner_state=InnerAgentGuardrailsGraphState(
                job_attachments=self._state_with_attachment().inner_state.job_attachments
            ),
        )

    @pytest.mark.asyncio
    async def test_tool_pre_node_judges_the_current_call_when_a_tool_is_called_twice(
        self, monkeypatch
    ):
        """One AI message, two calls to the same tool: after the first call's ToolMessage
        lands, the second evaluation must judge the second call's arguments and file,
        not the first call's (the tool node selects the call the same way)."""
        fake = _patch_uipath(monkeypatch, reason="ok")
        other = "00000000-0000-4000-8000-000000000001"
        first_args = {"attachment": {"ID": other}, "question": "first"}
        second_args = {"attachment": {"ID": self._UUID}, "question": "second"}
        state = AgentGuardrailsGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "my_tool", "args": first_args, "id": "c1"},
                        {"name": "my_tool", "args": second_args, "id": "c2"},
                    ],
                ),
                ToolMessage(content="first done", tool_call_id="c1"),
            ],
            inner_state=InnerAgentGuardrailsGraphState(
                job_attachments=self._state_with_attachment().inner_state.job_attachments
            ),
        )

        _, node = create_tool_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        cmd = await node(state)

        assert cmd.goto == "ok"
        assert json.loads(fake.guardrails.last_text) == second_args
        assert [a.id for a in fake.guardrails.last_attachments] == [self._UUID]

    @pytest.mark.asyncio
    async def test_tool_post_node_judges_the_answered_call_when_a_tool_is_called_twice(
        self, monkeypatch
    ):
        """After the second of two calls returns, the post evaluation reads that
        call's result (the last ToolMessage), which here names the file."""
        fake = _patch_uipath(monkeypatch, reason="ok")
        second_result = json.dumps(
            {
                "file": {
                    "ID": self._UUID,
                    "FullName": "Tickets.csv",
                    "MimeType": "text/csv",
                }
            }
        )
        state = AgentGuardrailsGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "my_tool", "args": {"n": 1}, "id": "c1"},
                        {"name": "my_tool", "args": {"n": 2}, "id": "c2"},
                    ],
                ),
                ToolMessage(content="first done", tool_call_id="c1"),
                ToolMessage(content=second_result, tool_call_id="c2"),
            ],
            inner_state=InnerAgentGuardrailsGraphState(
                job_attachments=self._state_with_attachment().inner_state.job_attachments
            ),
        )

        _, node = create_tool_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.POST_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        await node(state)

        assert fake.guardrails.last_text == second_result
        assert [a.id for a in fake.guardrails.last_attachments] == [self._UUID]

    @pytest.mark.asyncio
    async def test_tool_pre_node_forwards_attachments_referenced_in_tool_args(
        self, monkeypatch
    ):
        """Before the tool runs, the judge reads the file the call names. The model
        passes ``{"ID": ...}`` only, so name and type come from the run's registry."""
        fake = _patch_uipath(monkeypatch, reason="ok")
        args = {"attachment": {"ID": self._UUID}, "question": "summarize"}

        _, node = create_tool_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        cmd = await node(self._tool_pre_state(args))

        assert cmd.goto == "ok"
        assert json.loads(fake.guardrails.last_text) == args
        assert [
            a.model_dump(by_alias=True) for a in fake.guardrails.last_attachments
        ] == [{"id": self._UUID, "fileName": "Tickets.csv", "mimeType": "text/csv"}]

    @pytest.mark.asyncio
    async def test_tool_pre_node_ignores_registry_when_args_reference_nothing(
        self, monkeypatch
    ):
        """A tool call without a file forwards nothing even though the run holds one;
        otherwise every tool call would ship every file to the backend."""
        fake = _patch_uipath(monkeypatch, reason="ok")

        _, node = create_tool_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        await node(self._tool_pre_state({"q": 1}))

        assert fake.guardrails.last_attachments == []

    @pytest.mark.asyncio
    async def test_tool_post_node_forwards_attachment_returned_by_the_tool(
        self, monkeypatch
    ):
        """After the tool runs, the judge reads the file the result names."""
        fake = _patch_uipath(monkeypatch, reason="ok")
        content = json.dumps(
            {
                "file": {
                    "ID": self._UUID,
                    "FullName": "Tickets.csv",
                    "MimeType": "text/csv",
                }
            }
        )

        _, node = create_tool_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.POST_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        await node(self._tool_post_state(content))

        assert fake.guardrails.last_text == content
        assert [a.id for a in fake.guardrails.last_attachments] == [self._UUID]

    @pytest.mark.asyncio
    async def test_tool_post_node_with_plain_text_result_forwards_nothing(
        self, monkeypatch
    ):
        fake = _patch_uipath(monkeypatch, reason="ok")

        _, node = create_tool_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.POST_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        await node(self._tool_post_state("tool output"))

        assert fake.guardrails.last_attachments == []

    @pytest.mark.asyncio
    async def test_tool_node_without_extractors_forwards_nothing(self, monkeypatch):
        """A tool-scope built-in node built without the argument/result extractors has
        no source to scan and must not fall back to the whole registry."""
        from uipath.platform.guardrails import GuardrailScope

        fake = _patch_uipath(monkeypatch, reason="ok")

        _, node = _create_guardrail_node(
            self._judge_guardrail(),
            GuardrailScope.TOOL,
            ExecutionStage.PRE_EXECUTION,
            lambda state: "payload",
            "ok",
            "nope",
        )
        cmd = await node(self._tool_pre_state({"attachment": {"ID": self._UUID}}))

        assert cmd.goto == "ok"
        assert fake.guardrails.last_attachments == []

    @pytest.mark.asyncio
    async def test_tool_post_node_evaluates_without_files_when_the_result_cannot_be_read(
        self, monkeypatch
    ):
        """A file must never fail the run: if the result extractor blows up, the judge
        still sees the text payload, just without attachments."""
        fake = _patch_uipath(monkeypatch, reason="ok")

        def broken(_state):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes._extract_tool_output_data",
            broken,
        )

        _, node = create_tool_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.POST_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        cmd = await node(self._tool_post_state('{"ID": "not-json-but-mentions-ID"'))

        assert cmd.goto == "ok"
        assert fake.guardrails.last_text == '{"ID": "not-json-but-mentions-ID"'
        assert fake.guardrails.last_attachments == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "factory", [create_agent_init_guardrail_node, create_llm_guardrail_node]
    )
    async def test_agent_and_llm_pre_nodes_forward_files_scoped_attachments(
        self, monkeypatch, factory
    ):
        """Pre-execution at Agent and LLM scope with ``appliesTo = Files`` still hands
        the run's files to the judge (real resolver, nothing patched)."""
        from uipath.platform.guardrails.guardrails import EnumParameterValue

        fake = _patch_uipath(monkeypatch, reason="ok")
        guardrail = self._judge_guardrail()
        guardrail.validator_parameters = [
            EnumParameterValue.model_validate(
                {"$parameterType": "enum", "id": "appliesTo", "value": "Files"}
            )
        ]

        _, node = factory(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )
        await node(self._state_with_attachment())

        assert [a.id for a in fake.guardrails.last_attachments] == [self._UUID]

    @pytest.mark.asyncio
    async def test_attachment_rejection_falls_back_to_text_only(self, monkeypatch):
        """A 400 on a request that carried attachments must not kill the run: the backend
        rejected the file references, so evaluate the text payload alone."""
        import httpx
        from uipath.platform.errors import EnrichedException
        from uipath.platform.guardrails import GuardrailAttachment

        calls: list[list[GuardrailAttachment] | None] = []
        response = httpx.Response(
            400, request=httpx.Request("POST", "https://x/validate"), text="bad url"
        )
        rejection = EnrichedException(
            httpx.HTTPStatusError("400", request=response.request, response=response)
        )

        class FlakyGuardrails:
            def evaluate_guardrail(
                self, text, guardrail, *, attachments=None, termination_mode=None
            ):
                calls.append(attachments)
                if attachments:
                    raise rejection
                return GuardrailValidationResult(
                    result=GuardrailValidationResultType.PASSED, reason="ok"
                )

        class FlakyUiPath:
            guardrails = FlakyGuardrails()

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath",
            lambda: FlakyUiPath(),
        )
        attachment = GuardrailAttachment(
            id=self._UUID,
            file_name="a.csv",
            mime_type="text/csv",
        )
        self._patch_resolver(monkeypatch, [attachment])

        _, node = create_agent_init_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )
        cmd = await node(self._state_with_attachment())

        assert cmd.goto == "ok"
        assert calls == [[attachment], None]

    @pytest.mark.asyncio
    async def test_non_attachment_400_still_propagates(self, monkeypatch):
        """Only an attachment-caused 400 is absorbed; a 400 without attachments is real."""
        import httpx
        from uipath.platform.errors import EnrichedException

        response = httpx.Response(
            400, request=httpx.Request("POST", "https://x/validate"), text="bad"
        )
        rejection = EnrichedException(
            httpx.HTTPStatusError("400", request=response.request, response=response)
        )

        class FailingGuardrails:
            def evaluate_guardrail(
                self, text, guardrail, *, attachments=None, termination_mode=None
            ):
                raise rejection

        class FailingUiPath:
            guardrails = FailingGuardrails()

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath",
            lambda: FailingUiPath(),
        )
        self._patch_resolver(monkeypatch, [])

        _, node = create_agent_init_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        with pytest.raises(EnrichedException):
            await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

    @pytest.mark.asyncio
    async def test_payload_generator_runs_once_per_evaluation(self, monkeypatch):
        """Regression guard: the generator used to run twice — once for observability
        metadata and once inside the evaluator — which would double every resolution."""
        calls = []
        _patch_uipath(monkeypatch)
        self._patch_resolver(monkeypatch, [])

        def counting_get_message_content(msg):
            calls.append(1)
            return "payload"

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.get_message_content",
            counting_get_message_content,
        )

        _, node = create_agent_init_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )
        await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_node_still_passes_when_no_attachment_resolved(self, monkeypatch):
        """The low-code node is fail-closed, so resolution must absorb its own errors."""
        fake = _patch_uipath(monkeypatch, reason="ok")
        self._patch_resolver(monkeypatch, [])

        _, node = create_agent_init_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )

        cmd = await node(self._state_with_attachment())

        assert cmd.goto == "ok"
        assert fake.guardrails.last_attachments == []


class TestGuardrailFlaggedFileNames:
    """The evaluation node writes the names of the files the guardrail service flagged
    to ``inner_state.guardrail_flagged_file_names`` on every outcome."""

    _CSV = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"
    _PDF = "0b6f3a2d-5e4c-4b1a-8f9e-1d2c3b4a5f60"

    @classmethod
    def _references(cls):
        from uipath.platform.guardrails import GuardrailAttachment

        return [
            GuardrailAttachment(id=cls._CSV, file_name="a.csv", mime_type="text/csv"),
            GuardrailAttachment(
                id=cls._PDF, file_name="b.pdf", mime_type="application/pdf"
            ),
        ]

    @staticmethod
    def _result(result, flagged_ids):
        validation = GuardrailValidationResult.model_validate(
            {"result": result, "reason": "r", "flaggedAttachmentIds": flagged_ids}
        )
        assert validation.flagged_attachment_ids == flagged_ids, (
            "uipath-core predates GuardrailValidationResult.flagged_attachment_ids"
        )
        return validation

    @classmethod
    def _patch(cls, monkeypatch, result, flagged_ids, references):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath",
            lambda: FakeUiPath(cls._result(result, flagged_ids)),
        )
        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.resolve_guardrail_attachments",
            AsyncMock(return_value=references),
        )

    @staticmethod
    async def _run_node():
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "Example"
        _, node = create_agent_init_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )
        return await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

    @pytest.mark.asyncio
    async def test_failure_writes_the_flagged_names_matching_ids_case_insensitively(
        self, monkeypatch
    ):
        self._patch(
            monkeypatch,
            GuardrailValidationResultType.VALIDATION_FAILED,
            [self._PDF.upper()],
            self._references(),
        )

        cmd = await self._run_node()

        assert cmd.goto == "nope"
        assert cmd.update["inner_state"]["guardrail_flagged_file_names"] == ["b.pdf"]

    @pytest.mark.asyncio
    async def test_names_keep_the_order_sent(self, monkeypatch):
        self._patch(
            monkeypatch,
            GuardrailValidationResultType.VALIDATION_FAILED,
            [self._PDF, self._CSV],
            self._references(),
        )

        cmd = await self._run_node()

        assert cmd.update["inner_state"]["guardrail_flagged_file_names"] == [
            "a.csv",
            "b.pdf",
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("result", "flagged_ids"),
        [
            (GuardrailValidationResultType.PASSED, None),
            (GuardrailValidationResultType.VALIDATION_FAILED, None),
            (GuardrailValidationResultType.VALIDATION_FAILED, []),
        ],
    )
    async def test_no_flagged_ids_writes_none(self, monkeypatch, result, flagged_ids):
        self._patch(monkeypatch, result, flagged_ids, self._references())

        cmd = await self._run_node()

        assert "guardrail_flagged_file_names" in cmd.update["inner_state"]
        assert cmd.update["inner_state"]["guardrail_flagged_file_names"] is None

    @pytest.mark.asyncio
    async def test_ids_that_match_no_sent_reference_are_dropped(self, monkeypatch):
        self._patch(
            monkeypatch,
            GuardrailValidationResultType.VALIDATION_FAILED,
            ["11111111-2222-3333-4444-555555555555", "not-a-uuid", "", self._CSV],
            self._references(),
        )

        cmd = await self._run_node()

        assert cmd.update["inner_state"]["guardrail_flagged_file_names"] == ["a.csv"]

    @pytest.mark.asyncio
    async def test_flagged_ids_without_sent_references_write_none(self, monkeypatch):
        self._patch(
            monkeypatch,
            GuardrailValidationResultType.VALIDATION_FAILED,
            [self._CSV],
            [],
        )

        cmd = await self._run_node()

        assert cmd.update["inner_state"]["guardrail_flagged_file_names"] is None

    @pytest.mark.asyncio
    async def test_deterministic_guardrail_writes_none(self, monkeypatch):
        from uipath.core.guardrails import DeterministicGuardrail

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes._evaluate_deterministic_guardrail",
            lambda *args, **kwargs: self._result(
                GuardrailValidationResultType.VALIDATION_FAILED, [self._CSV]
            ),
        )
        guardrail = MagicMock(spec=DeterministicGuardrail)
        guardrail.name = "Deterministic"
        _, node = create_tool_guardrail_node(
            guardrail=guardrail,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
            tool_name="my_tool",
        )
        state = AgentGuardrailsGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "my_tool", "args": {"x": 1}, "id": "c1"}],
                )
            ]
        )

        cmd = await node(state)

        assert cmd.goto == "nope"
        assert "guardrail_flagged_file_names" in cmd.update["inner_state"]
        assert cmd.update["inner_state"]["guardrail_flagged_file_names"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("result", "flagged_ids", "expected"),
        [
            (GuardrailValidationResultType.PASSED, None, None),
            (GuardrailValidationResultType.VALIDATION_FAILED, [_PDF], ["b.pdf"]),
        ],
    )
    async def test_a_value_from_an_earlier_guardrail_never_survives_the_merge(
        self, monkeypatch, result, flagged_ids, expected
    ):
        from uipath_langchain.agent.react.reducers import merge_objects

        stale = InnerAgentGuardrailsGraphState(guardrail_flagged_file_names=["a.csv"])
        self._patch(monkeypatch, result, flagged_ids, self._references())

        cmd = await self._run_node()
        merged = merge_objects(stale, cmd.update["inner_state"])

        assert merged.guardrail_flagged_file_names == expected

    def test_state_without_the_key_loads(self):
        state = InnerAgentGuardrailsGraphState.model_validate(
            {
                "guardrail_validation_result": False,
                "guardrail_validation_details": "r",
                "guardrail_span_id": "span-1",
            }
        )

        assert state.guardrail_flagged_file_names is None


class TestGuardrailTerminationModeForwarding:
    """Each action's termination mode reaches the validate call, so the service stops
    scanning at the first violation for Block and Log and scans everything otherwise."""

    def test_actions_declare_their_termination_mode(self):
        from uipath.platform.guardrails import GuardrailTerminationMode

        from uipath_langchain.agent.guardrails.actions.block_action import BlockAction
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            EscalateAction,
        )
        from uipath_langchain.agent.guardrails.actions.filter_action import (
            FilterAction,
        )
        from uipath_langchain.agent.guardrails.actions.log_action import LogAction

        assert BlockAction("r").termination_mode == GuardrailTerminationMode.FAIL_FAST
        assert LogAction(None).termination_mode == GuardrailTerminationMode.FAIL_FAST
        assert FilterAction().termination_mode == GuardrailTerminationMode.EVALUATE_ALL
        assert (
            EscalateAction("app", None, 1, MagicMock()).termination_mode
            == GuardrailTerminationMode.EVALUATE_ALL
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["FailFast", "EvaluateAll", None])
    async def test_node_sends_the_termination_mode(self, monkeypatch, mode):
        from uipath.platform.guardrails import GuardrailTerminationMode

        termination_mode = GuardrailTerminationMode(mode) if mode else None
        fake = _patch_uipath(monkeypatch)
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "Example"
        _, node = create_agent_init_guardrail_node(
            guardrail,
            ExecutionStage.PRE_EXECUTION,
            "ok",
            "nope",
            termination_mode=termination_mode,
        )

        await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

        assert fake.guardrails.last_termination_mode == termination_mode

    @pytest.mark.asyncio
    async def test_text_only_retry_keeps_the_termination_mode(self, monkeypatch):
        import httpx
        from uipath.platform.errors import EnrichedException
        from uipath.platform.guardrails import (
            GuardrailAttachment,
            GuardrailTerminationMode,
        )

        from uipath_langchain.agent.guardrails.guardrail_nodes import (
            _evaluate_builtin_guardrail,
        )

        response = httpx.Response(
            400, request=httpx.Request("POST", "https://x/validate"), text="bad url"
        )
        rejection = EnrichedException(
            httpx.HTTPStatusError("400", request=response.request, response=response)
        )
        calls: list[tuple[Any, Any]] = []

        class FlakyGuardrails:
            def evaluate_guardrail(
                self, text, guardrail, *, attachments=None, termination_mode=None
            ):
                calls.append((attachments, termination_mode))
                if attachments:
                    raise rejection
                return GuardrailValidationResult(
                    result=GuardrailValidationResultType.PASSED, reason="ok"
                )

        class FlakyUiPath:
            guardrails = FlakyGuardrails()

        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath",
            lambda: FlakyUiPath(),
        )
        attachment = GuardrailAttachment(
            id="7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10",
            file_name="a.csv",
            mime_type="text/csv",
        )
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.name = "Example"

        await _evaluate_builtin_guardrail(
            guardrail, "payload", [attachment], GuardrailTerminationMode.FAIL_FAST
        )

        assert calls == [
            ([attachment], GuardrailTerminationMode.FAIL_FAST),
            (None, GuardrailTerminationMode.FAIL_FAST),
        ]

    def test_subgraph_passes_each_actions_termination_mode_to_its_node(self):
        from uipath.platform.guardrails import GuardrailScope, GuardrailTerminationMode

        from tests.agent.guardrails.test_guardrail_utils import FakeStateGraph
        from uipath_langchain.agent.guardrails.actions.block_action import BlockAction
        from uipath_langchain.agent.guardrails.actions.filter_action import (
            FilterAction,
        )
        from uipath_langchain.agent.guardrails.actions.log_action import LogAction
        from uipath_langchain.agent.react.guardrails import guardrails_subgraph

        seen: list[tuple[str, Any]] = []

        def factory(
            guardrail,
            execution_stage,
            success_node,
            failure_node,
            *,
            termination_mode=None,
        ):
            seen.append((guardrail.name, termination_mode))
            return f"eval_{guardrail.name}", (lambda s: s)

        guardrails = []
        for name, action in (
            ("blocking", BlockAction("r")),
            ("logging", LogAction(None)),
            ("filtering", FilterAction()),
        ):
            guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
            guardrail.name = name
            guardrails.append((guardrail, action))

        guardrails_subgraph._build_guardrail_node_chain(
            FakeStateGraph(None),  # type: ignore[arg-type]
            guardrails,
            GuardrailScope.LLM,
            ExecutionStage.PRE_EXECUTION,
            factory,
            "next",
            "inner",
        )

        assert sorted(seen) == [
            ("blocking", GuardrailTerminationMode.FAIL_FAST),
            ("filtering", GuardrailTerminationMode.EVALUATE_ALL),
            ("logging", GuardrailTerminationMode.FAIL_FAST),
        ]
