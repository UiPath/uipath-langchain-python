"""Tests for EscalateAction guardrail failure behavior."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from uipath.platform.guardrails import GuardrailScope
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.exceptions import AgentTerminationException
from uipath_langchain.agent.guardrails.actions.escalate_action import EscalateAction
from uipath_langchain.agent.guardrails.types import AgentGuardrailsGraphState


class TestEscalateAction:
    @pytest.mark.asyncio
    async def test_node_name_pre_llm(self):
        """PreExecution + LLM: name is sanitized correctly."""
        action = EscalateAction(
            app_name="TestApp",
            app_title="Test Title",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "My Guardrail v1"
        guardrail.description = "Test description"

        node_name, _ = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage="PreExecution",
        )

        assert node_name == "my_guardrail_v1_hitl_PreExecution_llm"

    @pytest.mark.asyncio
    async def test_node_name_post_agent(self):
        """PostExecution + AGENT: name is sanitized correctly."""
        action = EscalateAction(
            app_name="TestApp",
            app_title="Test Title",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Special-Guardrail@2024"
        guardrail.description = "Test description"

        node_name, _ = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.AGENT,
            execution_stage="PostExecution",
        )

        assert node_name == "special_guardrail_2024_hitl_PostExecution_agent"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_interrupts_with_correct_data_pre_llm(self, mock_interrupt):
        """PreExecution + LLM: interrupt is called with correct escalation data."""
        action = EscalateAction(
            app_name="TestApp",
            app_title="Test Title",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        # Mock interrupt to return approved escalation
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage="PreExecution",
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
            guardrail_validation_result="Validation failed",
        )

        await node(state)

        # Verify interrupt was called
        assert mock_interrupt.called
        call_args = mock_interrupt.call_args[0][0]

        # Verify escalation data structure
        assert call_args.app_name == "TestApp"
        assert call_args.app_folder_path == "TestFolder"
        assert call_args.title == "Test Title"
        assert call_args.assignee == "test@example.com"
        assert call_args.data["GuardrailName"] == "Test Guardrail"
        assert call_args.data["GuardrailDescription"] == "Test description"
        assert call_args.data["ExecutionStage"] == "PreExecution"
        assert call_args.data["GuardrailResult"] == "Validation failed"
        assert "ToolInputs" in call_args.data

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_approval_returns_command(self, mock_interrupt):
        """When escalation is approved, returns Command from _process_escalation_response."""
        action = EscalateAction(
            app_name="TestApp",
            app_title="Test Title",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        # Mock interrupt to return approved escalation
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage="PreExecution",
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        result = await node(state)

        # Should return Command or empty dict (from _process_escalation_response)
        assert isinstance(result, (Command, dict))

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_rejection_raises_exception(self, mock_interrupt):
        """When escalation is rejected, raises AgentTerminationException."""
        action = EscalateAction(
            app_name="TestApp",
            app_title="Test Title",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        # Mock interrupt to return rejected escalation
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Reject"
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage="PreExecution",
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(state)

        assert excinfo.value.error_info.title == "Escalation rejected"
        assert excinfo.value.error_info.detail == "Escalation rejected"
        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.CREATE_RESUME_TRIGGER_ERROR.value}"
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_post_execution_tool_field(self, mock_interrupt):
        """PostExecution: uses ToolOutputs field instead of ToolInputs."""
        action = EscalateAction(
            app_name="TestApp",
            app_title="Test Title",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        # Mock interrupt to return approved escalation
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage="PostExecution",
        )

        state = AgentGuardrailsGraphState(
            messages=[AIMessage(content="Test response")],
        )

        await node(state)

        # Verify ToolOutputs is used for PostExecution
        call_args = mock_interrupt.call_args[0][0]
        assert "ToolOutputs" in call_args.data
        assert "ToolInputs" not in call_args.data
