"""Tests for guardrail node creation and routing."""

import json
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

    def evaluate_guardrail(self, text, guardrail, *, attachments=None):
        self.call_count += 1
        self.last_text = text
        self.last_guardrail = guardrail
        self.last_attachments = attachments
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
            url="https://x/a.csv",
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
    """Agent- and LLM-scope nodes forward the run's job attachments to the judge."""

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
                        ID=self._UUID, FullName="Tickets.csv", MimeType="text/csv"
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
            url="https://x/Tickets.csv?sig=s",
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
            url="https://x/Tickets.csv?sig=s",
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

    @pytest.mark.asyncio
    async def test_span_metadata_never_carries_the_sas_url(self, monkeypatch):
        """The resolved url is a credential; only identity may reach observability."""
        from uipath.platform.guardrails import GuardrailAttachment

        _patch_uipath(monkeypatch)
        self._patch_resolver(
            monkeypatch,
            [
                GuardrailAttachment(
                    id=self._UUID,
                    file_name="Tickets.csv",
                    mime_type="text/csv",
                    url="https://x/Tickets.csv?sig=SECRET",
                )
            ],
        )

        _, node = create_agent_init_guardrail_node(
            guardrail=self._judge_guardrail(),
            execution_stage=ExecutionStage.PRE_EXECUTION,
            success_node="ok",
            failure_node="nope",
        )
        await node(self._state_with_attachment())

        metadata = node.__metadata__
        assert metadata["payload"]["attachments"] == [
            {"id": self._UUID, "fileName": "Tickets.csv", "mimeType": "text/csv"}
        ]
        assert "SECRET" not in json.dumps(metadata["payload"])

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
