"""Tests for EscalateAction guardrail failure behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from uipath.platform.guardrails import GuardrailScope
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.exceptions import AgentTerminationException
from uipath_langchain.agent.guardrails.actions.escalate_action import EscalateAction
from uipath_langchain.agent.guardrails.types import (
    ExecutionStage,
)
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState


class TestEscalateAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage", "expected_node_name"),
        [
            (
                GuardrailScope.LLM,
                ExecutionStage.PRE_EXECUTION,
                "llm_pre_execution_my_guardrail_1_hitl",
            ),
            (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
                "llm_post_execution_my_guardrail_1_hitl",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.PRE_EXECUTION,
                "agent_pre_execution_my_guardrail_1_hitl",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.POST_EXECUTION,
                "agent_post_execution_my_guardrail_1_hitl",
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.PRE_EXECUTION,
                "tool_pre_execution_my_guardrail_1_hitl",
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.POST_EXECUTION,
                "tool_post_execution_my_guardrail_1_hitl",
            ),
        ],
    )
    async def test_node_name(
        self, scope: GuardrailScope, stage: ExecutionStage, expected_node_name: str
    ) -> None:
        """Node name is sanitized correctly for each scope/stage."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "My Guardrail 1"
        guardrail.description = "Test description"

        node_name, _ = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=stage,
            guarded_component_name="test_node",
        )

        assert node_name == expected_node_name

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage", "expected_stage"),
        [
            (GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION, "PreExecution"),
            (GuardrailScope.LLM, ExecutionStage.POST_EXECUTION, "PostExecution"),
            (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION, "PreExecution"),
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_interrupts_with_correct_message_data(
        self,
        mock_interrupt,
        scope: GuardrailScope,
        stage: ExecutionStage,
        expected_stage: str,
    ) -> None:
        """Interrupt is called with correct escalation data for scope/stage."""
        action = EscalateAction(
            app_name="TestApp",
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
            scope=scope,
            execution_stage=stage,
            guarded_component_name="test_node",
        )

        if stage == ExecutionStage.PRE_EXECUTION:
            state = AgentGuardrailsGraphState(
                messages=[HumanMessage(content="Test message")],
                guardrail_validation_result="Validation failed",
            )
        else:
            # For POST_EXECUTION, LLM expects a prior input message + output message.
            state = AgentGuardrailsGraphState(
                messages=[
                    HumanMessage(content="Test message"),
                    HumanMessage(content="Output message"),
                ],
                guardrail_validation_result="Validation failed",
            )

        await node(state)

        # Verify interrupt was called
        assert mock_interrupt.called
        call_args = mock_interrupt.call_args[0][0]

        # Verify escalation data structure
        assert call_args.app_name == "TestApp"
        assert call_args.app_folder_path == "TestFolder"
        assert call_args.title == "Agents Guardrail Task"
        assert call_args.assignee == "test@example.com"
        assert call_args.data["GuardrailName"] == "Test Guardrail"
        assert call_args.data["GuardrailDescription"] == "Test description"
        assert call_args.data["ExecutionStage"] == expected_stage
        assert call_args.data["GuardrailResult"] == "Validation failed"

        if stage == ExecutionStage.PRE_EXECUTION:
            assert call_args.data["Inputs"] == "Test message"
            assert "Outputs" not in call_args.data
        else:
            assert call_args.data["Inputs"] == "Test message"
            assert call_args.data["Outputs"] == "Output message"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_post_agent_interrupts_with_correct_agent_result_data(
        self,
        mock_interrupt,
    ) -> None:
        """Interrupt is called with correct escalation data for scope/stage."""
        action = EscalateAction(
            app_name="TestApp",
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
            scope=GuardrailScope.AGENT,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_node",
        )

        state = AgentGuardrailsGraphState(
            messages=[
                HumanMessage(content="System prompt message"),
                HumanMessage(content="User prompt message"),
            ],
            agent_result={"ok": True},
            guardrail_validation_result="Validation failed",
        )

        await node(state)

        # Verify interrupt was called
        assert mock_interrupt.called
        call_args = mock_interrupt.call_args[0][0]

        # Verify escalation data structure
        assert call_args.app_name == "TestApp"
        assert call_args.app_folder_path == "TestFolder"
        assert call_args.title == "Agents Guardrail Task"
        assert call_args.assignee == "test@example.com"
        assert call_args.data["GuardrailName"] == "Test Guardrail"
        assert call_args.data["GuardrailDescription"] == "Test description"
        assert call_args.data["ExecutionStage"] == "PostExecution"
        assert call_args.data["GuardrailResult"] == "Validation failed"

        assert call_args.data["Inputs"] == "User prompt message"
        assert call_args.data["Outputs"] == '{"ok": true}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage"),
        [
            (
                GuardrailScope.LLM,
                ExecutionStage.PRE_EXECUTION,
            ),
            (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.PRE_EXECUTION,
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.POST_EXECUTION,
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.PRE_EXECUTION,
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.POST_EXECUTION,
            ),
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_approval_returns_command(
        self, mock_interrupt, scope: GuardrailScope, stage: ExecutionStage
    ):
        """When escalation is approved, returns Command from _process_escalation_response."""
        action = EscalateAction(
            app_name="TestApp",
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
            scope=scope,
            execution_stage=stage,
            guarded_component_name="test_node",
        )

        if stage == ExecutionStage.PRE_EXECUTION:
            state = AgentGuardrailsGraphState(
                messages=[HumanMessage(content="Test message")],
            )
        else:
            state = AgentGuardrailsGraphState(
                messages=[
                    HumanMessage(content="Test message"),
                    HumanMessage(content="Output message"),
                ],
            )

        result = await node(state)

        # Should return Command or empty dict (from _process_escalation_response)
        assert isinstance(result, (Command, dict))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage"),
        [
            (
                GuardrailScope.LLM,
                ExecutionStage.PRE_EXECUTION,
            ),
            (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.PRE_EXECUTION,
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.POST_EXECUTION,
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.PRE_EXECUTION,
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.POST_EXECUTION,
            ),
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_rejection_raises_exception(
        self, mock_interrupt, scope: GuardrailScope, stage: ExecutionStage
    ):
        """When escalation is rejected, raises AgentTerminationException."""
        action = EscalateAction(
            app_name="TestApp",
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
        mock_escalation_result.data = {"Reason": "Incorrect data"}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=stage,
            guarded_component_name="test_node",
        )

        if stage == ExecutionStage.PRE_EXECUTION:
            state = AgentGuardrailsGraphState(
                messages=[HumanMessage(content="Test message")],
            )
        else:
            state = AgentGuardrailsGraphState(
                messages=[
                    HumanMessage(content="Test message"),
                    HumanMessage(content="Output message"),
                ],
            )

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(state)

        assert excinfo.value.error_info.title == "Escalation rejected"
        assert (
            excinfo.value.error_info.detail
            == "Please contact your administrator. Action was rejected after reviewing the task created by guardrail [Test Guardrail], with reason: Incorrect data"
        )
        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scope",
        [
            GuardrailScope.LLM,
            GuardrailScope.AGENT,
            GuardrailScope.TOOL,
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_post_execution_single_message_raises_error(
        self, mock_interrupt, scope: GuardrailScope
    ):
        """PostExecution with only 1 message for LLM scope: raises AgentTerminationException."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        # Mock should not be called since validation happens before interrupt
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_node",
        )

        # Only one message in state - should raise error
        state = AgentGuardrailsGraphState(
            messages=[AIMessage(content="Only one message")],
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(state)

        # Verify error details
        assert excinfo.value.error_info.title == "Invalid state for POST_EXECUTION"
        assert "requires at least 2 messages" in excinfo.value.error_info.detail
        assert "found 1" in excinfo.value.error_info.detail
        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )

        # Verify interrupt was never called
        assert not mock_interrupt.called

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_post_execution_ai_message_with_tool_calls_extraction(
        self, mock_interrupt
    ):
        """PostExecution with AIMessage and tool calls: extracts tool calls and content."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_node",
        )

        ai_message = AIMessage(
            content="AI response",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"content": {"input": "test"}},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(
            messages=[
                HumanMessage(content="Input message"),
                ai_message,
            ]
        )

        await node(state)

        # Verify interrupt was called with tool calls (name and args) in Outputs and Inputs
        call_args = mock_interrupt.call_args[0][0]
        assert call_args.data["Inputs"] == "Input message"
        tool_outputs = call_args.data["Outputs"]
        parsed = json.loads(tool_outputs)
        assert len(parsed) == 1  # Tool call data with name and args
        assert parsed[0]["name"] == "test_tool"
        assert parsed[0]["args"] == {"content": {"input": "test"}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage"),
        [
            (
                GuardrailScope.LLM,
                ExecutionStage.PRE_EXECUTION,
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.PRE_EXECUTION,
            ),
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_pre_execution_with_reviewed_inputs(
        self, mock_interrupt, scope: GuardrailScope, stage: ExecutionStage
    ):
        """PreExecution: updates message content with ReviewedInputs."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_content = {"updated": "content"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_content)}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=stage,
            guarded_component_name="test_node",
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Original content")],
        )

        result = await node(state)

        # Verify message content was updated
        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["messages"][0].content == reviewed_content

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_post_execution_human_message_with_reviewed_outputs(
        self, mock_interrupt
    ):
        """PostExecution with HumanMessage: updates content with ReviewedOutputs."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_content = ["Updated content"]
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": json.dumps(reviewed_content)}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_node",
        )

        state = AgentGuardrailsGraphState(
            messages=[
                AIMessage(content="Previous AI message"),
                HumanMessage(content="Original content"),
            ],
        )

        result = await node(state)

        # Verify HumanMessage content was updated with fallback (raw JSON string)
        assert isinstance(result, Command)
        assert result.update is not None
        assert (
            result.update["messages"][0].content == "Previous AI message"
        )  # First message unchanged
        assert result.update["messages"][1].content == json.dumps(
            reviewed_content
        )  # Last message updated

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_post_execution_ai_message_with_reviewed_outputs_and_tool_calls(
        self, mock_interrupt
    ):
        """PostExecution with AIMessage: updates tool calls and content with ReviewedOutputs."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_tool_args = {"updated": "tool_content"}
        reviewed_outputs = [{"name": "test_tool", "args": reviewed_tool_args}]
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": json.dumps(reviewed_outputs)}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_node",
        )

        ai_message = AIMessage(
            content="Original content",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"content": {"original": "tool_input"}},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(
            messages=[
                HumanMessage(content="Previous input"),
                ai_message,
            ]
        )

        result = await node(state)

        # Verify tool calls args were updated by matching name
        assert isinstance(result, Command)
        assert result.update is not None
        assert (
            result.update["messages"][0].content == "Previous input"
        )  # First message unchanged
        updated_message = result.update["messages"][1]
        assert updated_message.tool_calls[0]["args"] == reviewed_tool_args

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_agent_post_execution_updates_agent_result(
        self, mock_interrupt
    ) -> None:
        """AGENT + POST_EXECUTION: updates agent_result using ReviewedOutputs."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_outputs = {"final": "approved"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": json.dumps(reviewed_outputs)}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.AGENT,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_node",
        )

        state = AgentGuardrailsGraphState(
            messages=[
                HumanMessage(content="Input message"),
                HumanMessage(content="Output message"),
            ],
            agent_result={"final": "original"},
        )

        result = await node(state)
        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["agent_result"] == reviewed_outputs

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_missing_reviewed_field_returns_empty_dict(self, mock_interrupt):
        """Missing reviewed field: returns empty dict when reviewed field not in result."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}  # No ReviewedInputs field
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_node",
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        result = await node(state)

        # Missing reviewed field should return empty dict
        assert result == {}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_json_parsing_error_raises_exception(self, mock_interrupt):
        """JSON parsing error: raises AgentTerminationException with execution error."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {
            "ReviewedInputs": "invalid json {"
        }  # Invalid JSON
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_node",
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(state)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Escalation rejected"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_interrupts_with_correct_data_pre_tool(self, mock_interrupt):
        """PreExecution + TOOL: interrupt is called with correct escalation data."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"input": "test"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(
            messages=[ai_message],
            guardrail_validation_result="Validation failed",
        )

        await node(state)

        assert mock_interrupt.called
        call_args = mock_interrupt.call_args[0][0]

        assert call_args.data["GuardrailName"] == "Test Guardrail"
        assert call_args.data["Component"] == "tool"
        assert call_args.data["ExecutionStage"] == "PreExecution"
        assert call_args.data["Inputs"] == '{"input": "test"}'

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_with_reviewed_inputs(self, mock_interrupt):
        """TOOL PreExecution: updates tool call arguments with ReviewedInputs."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_args = {"input": "updated_value", "param": "new"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_args)}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"input": "original_value"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        assert updated_message.tool_calls[0]["args"] == reviewed_args

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_post_execution_with_reviewed_outputs(self, mock_interrupt):
        """TOOL PostExecution: updates tool message content with ReviewedOutputs."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_output = "Updated tool output"
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": reviewed_output}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_tool",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"input": "test_value"},
                    "id": "call_1",
                }
            ],
        )
        tool_message = ToolMessage(
            content="Original tool output",
            tool_call_id="call_1",
        )
        state = AgentGuardrailsGraphState(messages=[ai_message, tool_message])

        result = await node(state)

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["messages"][0].content == ""  # First message unchanged
        updated_message = result.update["messages"][1]
        assert updated_message.content == reviewed_output

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_post_execution_extraction(self, mock_interrupt):
        """TOOL PostExecution: extracts tool message content correctly."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_tool",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "id": "call_1",
                }
            ],
        )
        tool_message = ToolMessage(
            content="Tool execution result",
            tool_call_id="call_1",
        )
        state = AgentGuardrailsGraphState(messages=[ai_message, tool_message])

        await node(state)

        call_args = mock_interrupt.call_args[0][0]
        assert (
            call_args.data["Inputs"] == '{"param": "value"}'
        )  # Extracted from AIMessage tool_calls
        assert call_args.data["Outputs"] == "Tool execution result"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_non_ai_message_returns_empty(
        self, mock_interrupt
    ):
        """TOOL PreExecution with non-AIMessage: returns empty dict."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps({})}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Not an AI message")],
        )

        result = await node(state)

        assert result == {}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_post_execution_non_tool_message_returns_empty(
        self, mock_interrupt
    ):
        """TOOL PostExecution with non-ToolMessage: returns empty dict."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": "test"}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="test_tool",
        )

        state = AgentGuardrailsGraphState(
            messages=[
                HumanMessage(content="Previous message"),
                AIMessage(content="Not a tool message"),
            ],
        )

        result = await node(state)

        assert result == {}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_empty_reviewed_inputs_returns_empty(
        self, mock_interrupt
    ):
        """TOOL PreExecution with empty ReviewedInputs: returns empty dict."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": ""}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"input": "test"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        assert result == {}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_non_dict_reviewed_inputs_raises_exception(
        self, mock_interrupt
    ):
        """TOOL PreExecution with invalid JSON ReviewedInputs: raises exception."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": "not a json dict"}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"input": "test"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(state)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Escalation rejected"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_json_error_raises_exception(self, mock_interrupt):
        """TOOL PreExecution with invalid JSON: raises AgentTerminationException."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": "invalid json {"}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"input": "test"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(state)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Escalation rejected"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_multiple_tool_calls(self, mock_interrupt):
        """TOOL PreExecution: updates multiple tool calls correctly."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_args = {"input": "updated_1"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_args)}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="tool_1",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tool_1",
                    "args": {"input": "original_1"},
                    "id": "call_1",
                },
                {
                    "name": "tool_2",
                    "args": {"input": "original_2"},
                    "id": "call_2",
                },
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        # Only tool_1 should be updated
        assert updated_message.tool_calls[0]["args"] == reviewed_args
        # tool_2 should remain unchanged
        assert updated_message.tool_calls[1]["args"] == {"input": "original_2"}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_fewer_reviewed_args_than_tool_calls(
        self, mock_interrupt
    ):
        """TOOL PreExecution: only updates tool calls with matching reviewed args."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            assignee="test@example.com",
        )
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"
        guardrail.description = "Test description"

        reviewed_args = {"input": "updated_1"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_args)}
        mock_interrupt.return_value = mock_escalation_result

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="tool_1",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tool_1",
                    "args": {"input": "original_1"},
                    "id": "call_1",
                },
                {
                    "name": "tool_2",
                    "args": {"input": "original_2"},
                    "id": "call_2",
                },
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        assert updated_message.tool_calls[0]["args"] == reviewed_args
        # Second tool call should remain unchanged
        assert updated_message.tool_calls[1]["args"] == {"input": "original_2"}

    @pytest.mark.asyncio
    async def test_extract_tool_content_pre_execution(self):
        """Extract TOOL content PreExecution: returns JSON array of tool call args."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_tool_escalation_content,
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"input": "test", "param": 123},
                    "id": "call_1",
                },
                {
                    "name": "another_tool",
                    "args": {"data": "value"},
                    "id": "call_2",
                },
            ],
        )

        result = _extract_tool_escalation_content(
            ai_message, ExecutionStage.PRE_EXECUTION, "test_tool"
        )

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"input": "test", "param": 123}

    @pytest.mark.asyncio
    async def test_extract_tool_content_post_execution(self):
        """Extract TOOL content PostExecution: returns tool message content."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_tool_escalation_content,
        )

        tool_message = ToolMessage(
            content="Tool execution result",
            tool_call_id="call_1",
        )

        result = _extract_tool_escalation_content(
            tool_message, ExecutionStage.POST_EXECUTION, "test_tool"
        )

        assert result == "Tool execution result"

    @pytest.mark.asyncio
    async def test_extract_tool_content_pre_execution_non_ai_message(self):
        """Extract TOOL content PreExecution with non-AIMessage: returns empty string."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_tool_escalation_content,
        )

        message = HumanMessage(content="Not an AI message")

        result = _extract_tool_escalation_content(
            message, ExecutionStage.PRE_EXECUTION, "test_tool"
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_tool_content_post_execution_non_tool_message(self):
        """Extract TOOL content PostExecution with non-ToolMessage: returns empty string."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_tool_escalation_content,
        )

        message = AIMessage(content="Not a tool message")

        result = _extract_tool_escalation_content(
            message, ExecutionStage.POST_EXECUTION, "test_tool"
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_tool_content_pre_execution_no_tool_calls(self):
        """Extract TOOL content PreExecution with no tool calls: returns empty array JSON."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_tool_escalation_content,
        )

        ai_message = AIMessage(content="No tool calls")

        result = _extract_tool_escalation_content(
            ai_message, ExecutionStage.PRE_EXECUTION, "test_tool"
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_llm_content_pre_execution_tool_message(self):
        """Extract LLM content PreExecution with ToolMessage: returns tool message content."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_llm_escalation_content,
        )

        tool_message = ToolMessage(
            content="Tool result",
            tool_call_id="call_1",
        )

        result = _extract_llm_escalation_content(
            tool_message, ExecutionStage.PRE_EXECUTION
        )

        assert result == "Tool result"

    @pytest.mark.asyncio
    async def test_extract_llm_content_pre_execution_empty_content(self):
        """Extract LLM content PreExecution with empty content: returns empty string."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_llm_escalation_content,
        )

        ai_message = AIMessage(content="")

        result = _extract_llm_escalation_content(
            ai_message, ExecutionStage.PRE_EXECUTION
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_llm_content_post_execution_tool_calls_no_content_field(self):
        """Extract LLM content PostExecution: extracts all tool calls with name and args."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _extract_llm_escalation_content,
        )

        ai_message = AIMessage(
            content="Response",
            tool_calls=[
                {
                    "name": "tool_without_content",
                    "args": {"param": "value"},  # No "content" field
                    "id": "call_1",
                }
            ],
        )

        result = _extract_llm_escalation_content(
            ai_message, ExecutionStage.POST_EXECUTION
        )

        assert isinstance(result, str)
        parsed = json.loads(result)
        # Should extract tool call data with name and args
        assert len(parsed) == 1
        assert parsed[0]["name"] == "tool_without_content"
        assert parsed[0]["args"] == {"param": "value"}

    @pytest.mark.asyncio
    async def test_validate_message_count_empty_messages_raises_exception(self):
        """Validate message count with empty messages: raises AgentTerminationException."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _validate_message_count,
        )

        state = AgentGuardrailsGraphState(messages=[])

        # Test PRE_EXECUTION validation
        with pytest.raises(AgentTerminationException) as excinfo:
            _validate_message_count(state, ExecutionStage.PRE_EXECUTION)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Invalid state for PRE_EXECUTION"
        assert "requires at least 1 message" in excinfo.value.error_info.detail

        # Test POST_EXECUTION validation
        with pytest.raises(AgentTerminationException) as excinfo:
            _validate_message_count(state, ExecutionStage.POST_EXECUTION)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Invalid state for POST_EXECUTION"
        assert "requires at least 2 messages" in excinfo.value.error_info.detail
