"""Tests for EscalateAction guardrail failure behavior."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from uipath.agent.models.agent import (
    AgentEscalationRecipientType,
    AssetRecipient,
    StandardRecipient,
)
from uipath.platform.action_center.tasks import Task, TaskRecipient, TaskRecipientType
from uipath.platform.guardrails import GuardrailScope
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.exceptions import AgentTerminationException
from uipath_langchain.agent.guardrails.actions.escalate_action import (
    EscalateAction,
    _deep_merge,
    _parse_reviewed_data,
)
from uipath_langchain.agent.guardrails.types import (
    ExecutionStage,
)
from uipath_langchain.agent.react.types import (
    AgentGuardrailsGraphState,
    InnerAgentGuardrailsGraphState,
)

DEFAULT_RECIPIENT = StandardRecipient(
    type=AgentEscalationRecipientType.USER_EMAIL,
    value="test@example.com",
)

STANDARD_USER_EMAIL_RECIPIENT = StandardRecipient(
    type=AgentEscalationRecipientType.USER_EMAIL,
    value="user@example.com",
)

STANDARD_USER_EMAIL_RECIPIENT_WITH_DISPLAY_NAME = StandardRecipient(
    type=AgentEscalationRecipientType.USER_EMAIL,
    value="user@example.com",
    display_name="John Doe",
)

STANDARD_GROUP_NAME_RECIPIENT = StandardRecipient(
    type=AgentEscalationRecipientType.GROUP_NAME,
    value="AdminGroup",
)

ASSET_USER_EMAIL_RECIPIENT = AssetRecipient(
    type=AgentEscalationRecipientType.ASSET_USER_EMAIL,
    asset_name="email_asset",
    folder_path="/Shared",
)

ASSET_GROUP_NAME_RECIPIENT = AssetRecipient(
    type=AgentEscalationRecipientType.ASSET_GROUP_NAME,
    asset_name="group_asset",
    folder_path="/Shared",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

MOCK_TASK_RECIPIENT = {"type": "UserEmail", "value": "test@example.com"}


def _make_mock_task_info(recipient: dict | None = None) -> dict:
    """Create a dict that can be passed to Task.model_validate()."""
    info: Dict[str, Any] = {
        "id": 123,
        "key": "test-key",
        "title": "Agents Guardrail Task",
    }
    if recipient is not None:
        info["recipient"] = recipient
    return info


def _make_mock_task(recipient: dict | None = None) -> Task:
    """Create a Task instance for tests."""
    info = _make_mock_task_info(recipient)
    if recipient:
        info["recipient"] = TaskRecipient.model_validate(recipient)
    return Task(**info)


def _get_action_nodes(action, guardrail, scope, stage, component_name="test_node"):
    """Call action_node and return (create_task_name, create_task_fn, interrupt_name, interrupt_fn)."""
    nodes = action.action_node(
        guardrail=guardrail,
        scope=scope,
        execution_stage=stage,
        guarded_component_name=component_name,
    )
    create_task_name, create_task_fn = nodes[0]
    interrupt_name, interrupt_fn = nodes[1]
    return create_task_name, create_task_fn, interrupt_name, interrupt_fn


def _make_state_with_task_info(
    messages,
    create_task_name,
    task_info=None,
    inner_state_kwargs=None,
):
    """Build an AgentGuardrailsGraphState with hitl_task_info pre-populated."""
    if task_info is None:
        task_info = _make_mock_task_info(recipient=MOCK_TASK_RECIPIENT)
    kwargs = inner_state_kwargs or {}
    return AgentGuardrailsGraphState(
        messages=messages,
        inner_state=InnerAgentGuardrailsGraphState(
            hitl_task_info={create_task_name: task_info},
            **kwargs,
        ),
    )


def _make_default_action():
    return EscalateAction(
        app_name="TestApp",
        app_folder_path="TestFolder",
        version=1,
        recipient=DEFAULT_RECIPIENT,
    )


def _make_default_guardrail(name="Test Guardrail", description="Test description"):
    guardrail = MagicMock()
    guardrail.name = name
    guardrail.description = description
    return guardrail


class TestEscalateAction:
    # ── Return type & node naming ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_action_node_returns_list_of_two_tuples(self) -> None:
        """action_node returns a list of exactly 2 (name, callable) tuples."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        nodes = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_node",
        )

        assert isinstance(nodes, list)
        assert len(nodes) == 2
        for name, fn in nodes:
            assert isinstance(name, str)
            assert callable(fn)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage", "expected_interrupt_name", "expected_create_task_name"),
        [
            (
                GuardrailScope.LLM,
                ExecutionStage.PRE_EXECUTION,
                "llm_pre_execution_my_guardrail_1_hitl",
                "llm_pre_execution_my_guardrail_1_hitl_create_task",
            ),
            (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
                "llm_post_execution_my_guardrail_1_hitl",
                "llm_post_execution_my_guardrail_1_hitl_create_task",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.PRE_EXECUTION,
                "agent_pre_execution_my_guardrail_1_hitl",
                "agent_pre_execution_my_guardrail_1_hitl_create_task",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.POST_EXECUTION,
                "agent_post_execution_my_guardrail_1_hitl",
                "agent_post_execution_my_guardrail_1_hitl_create_task",
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.PRE_EXECUTION,
                "tool_pre_execution_my_guardrail_1_hitl",
                "tool_pre_execution_my_guardrail_1_hitl_create_task",
            ),
            (
                GuardrailScope.TOOL,
                ExecutionStage.POST_EXECUTION,
                "tool_post_execution_my_guardrail_1_hitl",
                "tool_post_execution_my_guardrail_1_hitl_create_task",
            ),
        ],
    )
    async def test_node_names(
        self,
        scope: GuardrailScope,
        stage: ExecutionStage,
        expected_interrupt_name: str,
        expected_create_task_name: str,
    ) -> None:
        """Both create-task and interrupt node names are sanitized correctly."""
        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            recipient=DEFAULT_RECIPIENT,
        )
        guardrail = MagicMock()
        guardrail.name = "My Guardrail 1"
        guardrail.description = "Test description"

        nodes = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=stage,
            guarded_component_name="test_node",
        )

        create_task_name = nodes[0][0]
        interrupt_name = nodes[1][0]

        assert create_task_name == expected_create_task_name
        assert interrupt_name == expected_interrupt_name

    # ── Create-task node ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage", "expected_stage"),
        [
            (GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION, "PreExecution"),
            (GuardrailScope.LLM, ExecutionStage.POST_EXECUTION, "PostExecution"),
            (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION, "PreExecution"),
        ],
    )
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_node_sends_correct_data(
        self,
        mock_resolve_recipient,
        mock_uipath_class,
        mock_config,
        scope: GuardrailScope,
        stage: ExecutionStage,
        expected_stage: str,
    ) -> None:
        """Create-task node calls tasks.create_async with correct data fields."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="test@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        create_task_name, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, scope, stage
        )

        if stage == ExecutionStage.PRE_EXECUTION:
            state = AgentGuardrailsGraphState(
                messages=[HumanMessage(content="Test message")],
                inner_state=InnerAgentGuardrailsGraphState(
                    guardrail_validation_details="Validation failed"
                ),
            )
        else:
            state = AgentGuardrailsGraphState(
                messages=[
                    HumanMessage(content="Test message"),
                    HumanMessage(content="Output message"),
                ],
                inner_state=InnerAgentGuardrailsGraphState(
                    guardrail_validation_details="Validation failed"
                ),
            )

        result = await create_task_fn(state)

        # Verify tasks.create_async was called
        mock_client.tasks.create_async.assert_called_once()
        call_kwargs = mock_client.tasks.create_async.call_args[1]

        assert call_kwargs["title"] == "Agents Guardrail Task"
        assert call_kwargs["app_name"] == "TestApp"
        assert call_kwargs["app_folder_path"] == "TestFolder"

        data = call_kwargs["data"]
        assert data["GuardrailName"] == "Test Guardrail"
        assert data["GuardrailDescription"] == "Test description"
        assert data["ExecutionStage"] == expected_stage
        assert data["GuardrailResult"] == "Validation failed"
        assert data["TenantName"] == "TestTenant"

        if stage == ExecutionStage.PRE_EXECUTION:
            assert data["Inputs"] == '"Test message"'
            assert "Outputs" not in data
        else:
            assert data["Inputs"] == '"Test message"'
            assert data["Outputs"] == '"Output message"'

        # Verify state update includes task info
        assert "inner_state" in result
        assert create_task_name in result["inner_state"]["hitl_task_info"]

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_node_post_agent_with_agent_result(
        self,
        mock_resolve_recipient,
        mock_uipath_class,
        mock_config,
    ) -> None:
        """Create-task node for AGENT POST_EXECUTION sends agent_result in Outputs."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="test@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.AGENT, ExecutionStage.POST_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[
                HumanMessage(content="System prompt message"),
                HumanMessage(content="User prompt message"),
            ],
            inner_state=InnerAgentGuardrailsGraphState(
                agent_result={"ok": True},
                guardrail_validation_details="Validation failed",
            ),
        )

        await create_task_fn(state)

        call_kwargs = mock_client.tasks.create_async.call_args[1]
        data = call_kwargs["data"]

        assert data["Inputs"] == '"User prompt message"'
        assert data["Outputs"] == '{"ok": true}'
        assert data["ExecutionStage"] == "PostExecution"

    @pytest.mark.asyncio
    async def test_create_task_node_skips_if_task_already_exists(self) -> None:
        """Create-task node returns empty dict when task already exists in state."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        create_task_name, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        # State already has a task for this node
        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test")],
            create_task_name=create_task_name,
        )

        result = await create_task_fn(state)
        assert result == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scope",
        [
            GuardrailScope.LLM,
            GuardrailScope.AGENT,
            GuardrailScope.TOOL,
        ],
    )
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_node_post_execution_single_message_raises_error(
        self, mock_resolve_recipient, scope: GuardrailScope
    ):
        """Create-task for PostExecution with only 1 message: raises AgentTerminationException."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="test@example.com", type=TaskRecipientType.EMAIL
        )

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, scope, ExecutionStage.POST_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[AIMessage(content="Only one message")],
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await create_task_fn(state)

        assert excinfo.value.error_info.title == "Invalid state for POST_EXECUTION"
        assert "requires at least 2 messages" in excinfo.value.error_info.detail
        assert "found 1" in excinfo.value.error_info.detail
        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_node_tool_pre_execution_extracts_tool_args(
        self,
        mock_resolve_recipient,
        mock_uipath_class,
        mock_config,
    ):
        """Create-task for TOOL PRE_EXECUTION: extracts tool call args into Inputs."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="test@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="test_tool",
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
            inner_state=InnerAgentGuardrailsGraphState(
                guardrail_validation_details="Validation failed"
            ),
        )

        await create_task_fn(state)

        call_kwargs = mock_client.tasks.create_async.call_args[1]
        data = call_kwargs["data"]

        assert data["GuardrailName"] == "Test Guardrail"
        assert data["Component"] == "test_tool"
        assert data["ExecutionStage"] == "PreExecution"
        assert data["Inputs"] == '{"input": "test"}'

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_node_tool_post_execution_extracts_tool_content(
        self,
        mock_resolve_recipient,
        mock_uipath_class,
        mock_config,
    ):
        """Create-task for TOOL POST_EXECUTION: extracts tool message content into Outputs."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="test@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION,
            component_name="test_tool",
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
        state = AgentGuardrailsGraphState(
            messages=[ai_message, tool_message],
            inner_state=InnerAgentGuardrailsGraphState(
                guardrail_validation_details="Validation failed"
            ),
        )

        await create_task_fn(state)

        call_kwargs = mock_client.tasks.create_async.call_args[1]
        data = call_kwargs["data"]

        assert data["Inputs"] == '{"param": "value"}'
        assert data["Outputs"] == "Tool execution result"

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_node_post_execution_ai_message_with_tool_calls(
        self,
        mock_resolve_recipient,
        mock_uipath_class,
        mock_config,
    ):
        """Create-task for LLM POST_EXECUTION with AIMessage and tool calls extracts tool calls."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="test@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.POST_EXECUTION
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
            ],
            inner_state=InnerAgentGuardrailsGraphState(
                guardrail_validation_details="Validation failed"
            ),
        )

        await create_task_fn(state)

        call_kwargs = mock_client.tasks.create_async.call_args[1]
        data = call_kwargs["data"]

        assert data["Inputs"] == '"Input message"'
        tool_outputs = data["Outputs"]
        parsed_obj = json.loads(tool_outputs)
        parsed_list = parsed_obj["tool_calls"]
        assert len(parsed_list) == 1
        assert parsed_list[0]["name"] == "test_tool"
        assert parsed_list[0]["args"] == {"content": {"input": "test"}}

    # ── Interrupt node ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_interrupt_node_raises_if_no_task_info(self) -> None:
        """Interrupt node raises AgentTerminationException when no task in state."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        _, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        # State with empty hitl_task_info
        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test")],
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await interrupt_fn(state)

        assert excinfo.value.error_info.title == "Escalation task not found"
        assert "create-task node must run before" in excinfo.value.error_info.detail

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_interrupt_node_calls_with_wait_escalation(
        self, mock_interrupt
    ) -> None:
        """Interrupt node calls interrupt() with a WaitEscalation containing the task."""
        from uipath.platform.common import WaitEscalation

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        task_info = _make_mock_task_info(recipient=MOCK_TASK_RECIPIENT)
        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test")],
            create_task_name=create_task_name,
            task_info=task_info,
        )

        await interrupt_fn(state)

        mock_interrupt.assert_called_once()
        wait_escalation = mock_interrupt.call_args[0][0]
        assert isinstance(wait_escalation, WaitEscalation)
        assert wait_escalation.app_folder_path == "TestFolder"
        assert wait_escalation.action.id == 123

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_interrupt_node_without_recipient(self, mock_interrupt) -> None:
        """Interrupt node handles task info without recipient (recipient=None)."""
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        # No recipient in task info
        task_info = _make_mock_task_info(recipient=None)
        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test")],
            create_task_name=create_task_name,
            task_info=task_info,
        )

        await interrupt_fn(state)

        wait_escalation = mock_interrupt.call_args[0][0]
        # WaitEscalation doesn't store recipient as a model field
        assert wait_escalation.app_folder_path == "TestFolder"
        assert wait_escalation.action.id == 123

    # ── Approval / rejection ──────────────────────────────────────────────

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage"),
        [
            (GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION),
            (GuardrailScope.LLM, ExecutionStage.POST_EXECUTION),
            (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION),
            (GuardrailScope.AGENT, ExecutionStage.POST_EXECUTION),
            (GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION),
            (GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION),
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_approval_returns_command(
        self, mock_interrupt, scope: GuardrailScope, stage: ExecutionStage
    ):
        """When escalation is approved, returns Command from _process_escalation_response."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, scope, stage
        )

        if stage == ExecutionStage.PRE_EXECUTION:
            state = _make_state_with_task_info(
                messages=[HumanMessage(content="Test message")],
                create_task_name=create_task_name,
            )
        else:
            state = _make_state_with_task_info(
                messages=[
                    HumanMessage(content="Test message"),
                    HumanMessage(content="Output message"),
                ],
                create_task_name=create_task_name,
            )

        result = await interrupt_fn(state)

        # Should return Command or dict (from _process_escalation_response + cleanup)
        assert isinstance(result, (Command, dict))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage"),
        [
            (GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION),
            (GuardrailScope.LLM, ExecutionStage.POST_EXECUTION),
            (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION),
            (GuardrailScope.AGENT, ExecutionStage.POST_EXECUTION),
            (GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION),
            (GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION),
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_node_rejection_raises_exception(
        self, mock_interrupt, scope: GuardrailScope, stage: ExecutionStage
    ):
        """When escalation is rejected, raises AgentTerminationException."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Reject"
        mock_escalation_result.data = {"Reason": "Incorrect data"}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, scope, stage
        )

        if stage == ExecutionStage.PRE_EXECUTION:
            state = _make_state_with_task_info(
                messages=[HumanMessage(content="Test message")],
                create_task_name=create_task_name,
            )
        else:
            state = _make_state_with_task_info(
                messages=[
                    HumanMessage(content="Test message"),
                    HumanMessage(content="Output message"),
                ],
                create_task_name=create_task_name,
            )

        with pytest.raises(AgentTerminationException) as excinfo:
            await interrupt_fn(state)

        assert excinfo.value.error_info.title == "Escalation rejected"
        assert (
            excinfo.value.error_info.detail
            == "Action was rejected after reviewing the task created by guardrail [Test Guardrail], with reason: Incorrect data"
        )
        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )

    # ── Reviewed inputs / outputs processing ──────────────────────────────

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage"),
        [
            (GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION),
            (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION),
        ],
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_pre_execution_with_reviewed_inputs(
        self, mock_interrupt, scope: GuardrailScope, stage: ExecutionStage
    ):
        """PreExecution: updates message content with ReviewedInputs."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_content = {"updated": "content"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_content)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, scope, stage
        )

        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Original content")],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["messages"][0].content == reviewed_content

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_post_execution_human_message_with_reviewed_outputs(
        self, mock_interrupt
    ):
        """PostExecution with HumanMessage: updates content with ReviewedOutputs."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_content = ["Updated content"]
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": json.dumps(reviewed_content)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.POST_EXECUTION
        )

        state = _make_state_with_task_info(
            messages=[
                AIMessage(content="Previous AI message"),
                HumanMessage(content="Original content"),
            ],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["messages"][0].content == "Previous AI message"
        assert result.update["messages"][1].content == json.dumps(reviewed_content)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_post_execution_ai_message_with_reviewed_outputs_and_tool_calls(
        self, mock_interrupt
    ):
        """PostExecution with AIMessage: updates tool calls and content with ReviewedOutputs."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_tool_args = {"updated": "tool_content"}
        reviewed_outputs = {
            "tool_calls": [{"name": "test_tool", "args": reviewed_tool_args}]
        }
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": json.dumps(reviewed_outputs)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.POST_EXECUTION
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
        state = _make_state_with_task_info(
            messages=[
                HumanMessage(content="Previous input"),
                ai_message,
            ],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["messages"][0].content == "Previous input"
        updated_message = result.update["messages"][1]
        assert updated_message.tool_calls[0]["args"] == reviewed_tool_args

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_agent_post_execution_updates_agent_result(
        self, mock_interrupt
    ) -> None:
        """AGENT + POST_EXECUTION: updates agent_result using ReviewedOutputs."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_outputs = {"final": "approved"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": json.dumps(reviewed_outputs)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.AGENT, ExecutionStage.POST_EXECUTION
        )

        state = _make_state_with_task_info(
            messages=[
                HumanMessage(content="Input message"),
                HumanMessage(content="Output message"),
            ],
            create_task_name=create_task_name,
            inner_state_kwargs={"agent_result": {"final": "original"}},
        )

        result = await interrupt_fn(state)
        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["inner_state"]["agent_result"] == reviewed_outputs

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_missing_reviewed_field_returns_cleanup_dict(self, mock_interrupt):
        """Missing reviewed field: returns cleanup dict when reviewed field not in result."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}  # No ReviewedInputs field
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test message")],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        # Result contains cleanup update (hitl_task_info cleared)
        assert isinstance(result, dict)
        assert result["inner_state"]["hitl_task_info"][create_task_name] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_json_parsing_error_raises_exception(self, mock_interrupt):
        """JSON parsing error: raises AgentTerminationException with execution error."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {
            "ReviewedInputs": "invalid json {"
        }  # Invalid JSON
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test message")],
            create_task_name=create_task_name,
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await interrupt_fn(state)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Escalation rejected"

    # ── TOOL scope interrupt processing ───────────────────────────────────

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_with_reviewed_inputs(self, mock_interrupt):
        """TOOL PreExecution: updates tool call arguments with ReviewedInputs."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_args = {"input": "updated_value", "param": "new"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_args)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="test_tool",
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
        state = _make_state_with_task_info(
            messages=[ai_message],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        assert updated_message.tool_calls[0]["args"] == reviewed_args

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_post_execution_with_reviewed_outputs(self, mock_interrupt):
        """TOOL PostExecution: updates tool message content with ReviewedOutputs."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_output = "Updated tool output"
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": reviewed_output}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION,
            component_name="test_tool",
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
        state = _make_state_with_task_info(
            messages=[ai_message, tool_message],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        assert result.update is not None
        assert result.update["messages"][0].content == ""
        updated_message = result.update["messages"][1]
        assert updated_message.content == reviewed_output

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_non_ai_message_returns_cleanup(
        self, mock_interrupt
    ):
        """TOOL PreExecution with non-AIMessage: returns cleanup dict."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps({})}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="test_tool",
        )

        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Not an AI message")],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        # Returns cleanup dict since no AI message found
        assert isinstance(result, dict)
        assert result["inner_state"]["hitl_task_info"][create_task_name] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_post_execution_non_tool_message_returns_cleanup(
        self, mock_interrupt
    ):
        """TOOL PostExecution with non-ToolMessage: returns cleanup dict."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedOutputs": "test"}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION,
            component_name="test_tool",
        )

        state = _make_state_with_task_info(
            messages=[
                HumanMessage(content="Previous message"),
                AIMessage(content="Not a tool message"),
            ],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, dict)
        assert result["inner_state"]["hitl_task_info"][create_task_name] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_empty_reviewed_inputs_returns_cleanup(
        self, mock_interrupt
    ):
        """TOOL PreExecution with empty ReviewedInputs: returns cleanup dict."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": ""}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="test_tool",
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
        state = _make_state_with_task_info(
            messages=[ai_message],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, dict)
        assert result["inner_state"]["hitl_task_info"][create_task_name] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_non_dict_reviewed_inputs_raises_exception(
        self, mock_interrupt
    ):
        """TOOL PreExecution with invalid JSON ReviewedInputs: raises exception."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": "not a json dict"}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="test_tool",
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
        state = _make_state_with_task_info(
            messages=[ai_message],
            create_task_name=create_task_name,
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await interrupt_fn(state)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Escalation rejected"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_json_error_raises_exception(self, mock_interrupt):
        """TOOL PreExecution with invalid JSON: raises AgentTerminationException."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": "invalid json {"}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="test_tool",
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
        state = _make_state_with_task_info(
            messages=[ai_message],
            create_task_name=create_task_name,
        )

        with pytest.raises(AgentTerminationException) as excinfo:
            await interrupt_fn(state)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Escalation rejected"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_multiple_tool_calls(self, mock_interrupt):
        """TOOL PreExecution: updates only matching tool call correctly."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_args = {"input": "updated_1"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_args)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="tool_1",
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
        state = _make_state_with_task_info(
            messages=[ai_message],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        assert updated_message.tool_calls[0]["args"] == reviewed_args
        assert updated_message.tool_calls[1]["args"] == {"input": "original_2"}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_tool_pre_execution_fewer_reviewed_args_than_tool_calls(
        self, mock_interrupt
    ):
        """TOOL PreExecution: only updates tool calls with matching reviewed args."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_args = {"input": "updated_1"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_args)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION,
            component_name="tool_1",
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
        state = _make_state_with_task_info(
            messages=[ai_message],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        assert updated_message.tool_calls[0]["args"] == reviewed_args
        assert updated_message.tool_calls[1]["args"] == {"input": "original_2"}

    # ── Cleanup of hitl_task_info ─────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_interrupt_node_cleans_up_task_info_on_approval(self, mock_interrupt):
        """On approval, the cleanup dict merges hitl_task_info=None into the result."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        reviewed_content = {"updated": "content"}
        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {"ReviewedInputs": json.dumps(reviewed_content)}
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test")],
            create_task_name=create_task_name,
        )

        result = await interrupt_fn(state)

        assert isinstance(result, Command)
        # Cleanup should set the task info to None
        assert result.update["inner_state"]["hitl_task_info"][create_task_name] is None

    # ── Metadata stored in escalation_data ────────────────────────────────

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_interrupt_node_stores_reviewed_data_in_metadata(self, mock_interrupt):
        """Interrupt node stores reviewed inputs/outputs/reason in metadata."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {
            "ReviewedInputs": json.dumps({"updated": "input"}),
            "ReviewedOutputs": json.dumps({"updated": "output"}),
            "Reason": "Looks good",
        }
        mock_escalation_result.completed_by_user = {
            "displayName": "Jane Doe",
            "emailAddress": "jane@example.com",
        }
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test")],
            create_task_name=create_task_name,
        )

        await interrupt_fn(state)

        metadata = getattr(interrupt_fn, "__metadata__", None)
        assert metadata is not None
        assert metadata["escalation_data"]["reviewed_inputs"] == {"updated": "input"}
        assert metadata["escalation_data"]["reviewed_outputs"] == {"updated": "output"}
        assert metadata["escalation_data"]["reason"] == "Looks good"
        assert metadata["escalation_data"]["reviewed_by"] == "Jane Doe"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    async def test_interrupt_node_reviewed_by_falls_back_to_email(self, mock_interrupt):
        """Reviewed_by falls back to emailAddress when displayName is missing."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        mock_escalation_result = MagicMock()
        mock_escalation_result.action = "Approve"
        mock_escalation_result.data = {}
        mock_escalation_result.completed_by_user = {
            "emailAddress": "jane@example.com",
        }
        mock_interrupt.return_value = mock_escalation_result

        create_task_name, _, _, interrupt_fn = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = _make_state_with_task_info(
            messages=[HumanMessage(content="Test")],
            create_task_name=create_task_name,
        )

        await interrupt_fn(state)

        metadata = getattr(interrupt_fn, "__metadata__", None)
        assert metadata["escalation_data"]["reviewed_by"] == "jane@example.com"

    # ── Content extraction (pure function tests – no change needed) ───────

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

        assert result == '""'

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
                    "args": {"param": "value"},
                    "id": "call_1",
                }
            ],
        )

        result = _extract_llm_escalation_content(
            ai_message, ExecutionStage.POST_EXECUTION
        )

        assert isinstance(result, str)
        parsed_obj = json.loads(result)
        parsed_list = parsed_obj["tool_calls"]
        assert len(parsed_list) == 1
        assert parsed_list[0]["name"] == "tool_without_content"
        assert parsed_list[0]["args"] == {"param": "value"}

    @pytest.mark.asyncio
    async def test_validate_message_count_empty_messages_raises_exception(self):
        """Validate message count with empty messages: raises AgentTerminationException."""
        from uipath_langchain.agent.guardrails.actions.escalate_action import (
            _validate_message_count,
        )

        state = AgentGuardrailsGraphState(messages=[])

        with pytest.raises(AgentTerminationException) as excinfo:
            _validate_message_count(state, ExecutionStage.PRE_EXECUTION)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Invalid state for PRE_EXECUTION"
        assert "requires at least 1 message" in excinfo.value.error_info.detail

        with pytest.raises(AgentTerminationException) as excinfo:
            _validate_message_count(state, ExecutionStage.POST_EXECUTION)

        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Invalid state for POST_EXECUTION"
        assert "requires at least 2 messages" in excinfo.value.error_info.detail

    # ── Recipient resolution ──────────────────────────────────────────────

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "recipient,expected_value",
        [
            (
                STANDARD_USER_EMAIL_RECIPIENT,
                TaskRecipient(value="user@example.com", type=TaskRecipientType.EMAIL),
            ),
            (
                STANDARD_GROUP_NAME_RECIPIENT,
                TaskRecipient(value="AdminGroup", type=TaskRecipientType.GROUP_NAME),
            ),
            (
                ASSET_USER_EMAIL_RECIPIENT,
                TaskRecipient(value="user@example.com", type=TaskRecipientType.EMAIL),
            ),
            (
                ASSET_GROUP_NAME_RECIPIENT,
                TaskRecipient(value="AdminGroup", type=TaskRecipientType.GROUP_NAME),
            ),
        ],
    )
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_resolves_recipient_correctly(
        self, mock_resolve_recipient, mock_uipath_class, mock_config, recipient, expected_value
    ) -> None:
        """Create-task node resolves recipient correctly for different types."""
        mock_resolve_recipient.return_value = expected_value
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            recipient=recipient,
        )
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        await create_task_fn(state)

        mock_resolve_recipient.assert_called_once_with(recipient)
        call_kwargs = mock_client.tasks.create_async.call_args[1]
        assert call_kwargs["recipient"] == expected_value

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_with_asset_recipient_resolution_failure(
        self, mock_resolve_recipient
    ) -> None:
        """Create-task with AssetRecipient: propagates asset resolution errors."""
        mock_resolve_recipient.side_effect = ValueError("Asset 'email_asset' not found")

        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            recipient=ASSET_USER_EMAIL_RECIPIENT,
        )
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        with pytest.raises(ValueError) as excinfo:
            await create_task_fn(state)

        assert "Asset 'email_asset' not found" in str(excinfo.value)


class TestEscalateActionMetadata:
    """Tests for EscalateAction node metadata."""

    @pytest.mark.asyncio
    async def test_both_nodes_share_same_metadata(self):
        """Both create-task and interrupt nodes share the same metadata object."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        nodes = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_node",
        )

        create_task_fn = nodes[0][1]
        interrupt_fn = nodes[1][1]

        create_metadata = getattr(create_task_fn, "__metadata__", None)
        interrupt_metadata = getattr(interrupt_fn, "__metadata__", None)

        assert create_metadata is not None
        assert interrupt_metadata is not None
        # They should be the exact same dict object
        assert create_metadata is interrupt_metadata

    @pytest.mark.asyncio
    async def test_metadata_has_escalation_data_with_recipient_type(self):
        """Metadata includes recipient_type from the action's recipient."""
        action = _make_default_action()
        guardrail = _make_default_guardrail()

        nodes = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_node",
        )

        metadata = getattr(nodes[0][1], "__metadata__", None)
        assert metadata is not None
        assert "escalation_data" in metadata
        assert metadata["escalation_data"]["recipient_type"] == "UserEmail"

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_standard_recipient_assigned_to_uses_value(
        self, mock_resolve_recipient, mock_interrupt, mock_uipath_class, mock_config
    ):
        """Test that assigned_to uses value for StandardRecipient without display_name."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="user@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            recipient=STANDARD_USER_EMAIL_RECIPIENT,
        )
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        await create_task_fn(state)

        metadata = getattr(create_task_fn, "__metadata__", None)
        assert metadata is not None
        assert metadata["escalation_data"]["assigned_to"] == "user@example.com"

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_standard_recipient_assigned_to_uses_display_name(
        self, mock_resolve_recipient, mock_interrupt, mock_uipath_class, mock_config
    ):
        """Test that assigned_to uses display_name when present for StandardRecipient."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="user@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            recipient=STANDARD_USER_EMAIL_RECIPIENT_WITH_DISPLAY_NAME,
        )
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        await create_task_fn(state)

        metadata = getattr(create_task_fn, "__metadata__", None)
        assert metadata is not None
        assert metadata["escalation_data"]["assigned_to"] == "John Doe"

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.interrupt")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_asset_recipient_assigned_to_uses_resolved_value(
        self, mock_resolve_recipient, mock_interrupt, mock_uipath_class, mock_config
    ):
        """Test that assigned_to uses resolved task_recipient value for AssetRecipient."""
        resolved_recipient = TaskRecipient(
            value="resolved@example.com", type=TaskRecipientType.EMAIL
        )
        mock_resolve_recipient.return_value = resolved_recipient
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = EscalateAction(
            app_name="TestApp",
            app_folder_path="TestFolder",
            version=1,
            recipient=ASSET_USER_EMAIL_RECIPIENT,
        )
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        await create_task_fn(state)

        metadata = getattr(create_task_fn, "__metadata__", None)
        assert metadata is not None
        assert metadata["escalation_data"]["assigned_to"] == "resolved@example.com"

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.UiPathConfig"
    )
    @patch("uipath_langchain.agent.guardrails.actions.escalate_action.UiPath")
    @patch(
        "uipath_langchain.agent.guardrails.actions.escalate_action.resolve_recipient_value"
    )
    async def test_create_task_node_sets_metadata_node_type(
        self, mock_resolve_recipient, mock_uipath_class, mock_config
    ):
        """Create-task node sets node_type='create_hitl_task' in metadata."""
        mock_resolve_recipient.return_value = TaskRecipient(
            value="test@example.com", type=TaskRecipientType.EMAIL
        )
        mock_config.base_url = None
        mock_config.tenant_name = "TestTenant"

        mock_task = _make_mock_task(recipient=MOCK_TASK_RECIPIENT)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=mock_task)
        mock_uipath_class.return_value = mock_client

        action = _make_default_action()
        guardrail = _make_default_guardrail()

        _, create_task_fn, _, _ = _get_action_nodes(
            action, guardrail, GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION
        )

        state = AgentGuardrailsGraphState(
            messages=[HumanMessage(content="Test message")],
        )

        await create_task_fn(state)

        metadata = getattr(create_task_fn, "__metadata__", None)
        assert metadata is not None
        assert metadata["node_type"] == "create_hitl_task"


class TestHelperFunctions:
    """Tests for helper functions: _deep_merge and _parse_reviewed_data."""

    # ── _deep_merge ───────────────────────────────────────────────────────

    def test_deep_merge_flat_dicts(self):
        """Merge flat dicts: source values overwrite target."""
        target = {"a": 1, "b": 2}
        source = {"b": 3, "c": 4}
        _deep_merge(target, source)
        assert target == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested_dicts(self):
        """Merge nested dicts: inner dicts are merged recursively."""
        target = {"outer": {"a": 1, "b": 2}}
        source = {"outer": {"b": 3, "c": 4}}
        _deep_merge(target, source)
        assert target == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deep_merge_mixed_types(self):
        """Merge with mixed types: non-dict values overwrite."""
        target = {"a": {"nested": True}, "b": "string"}
        source = {"a": "overwritten", "c": {"new": True}}
        _deep_merge(target, source)
        assert target == {"a": "overwritten", "b": "string", "c": {"new": True}}

    def test_deep_merge_empty_source(self):
        """Merge empty source: target unchanged."""
        target = {"a": 1}
        _deep_merge(target, {})
        assert target == {"a": 1}

    def test_deep_merge_empty_target(self):
        """Merge into empty target: target becomes source."""
        target: Dict[str, Any] = {}
        _deep_merge(target, {"a": 1})
        assert target == {"a": 1}

    def test_deep_merge_deeply_nested(self):
        """Three-level deep merge."""
        target = {"l1": {"l2": {"l3": "old", "keep": True}}}
        source = {"l1": {"l2": {"l3": "new", "add": True}}}
        _deep_merge(target, source)
        assert target == {"l1": {"l2": {"l3": "new", "keep": True, "add": True}}}

    # ── _parse_reviewed_data ──────────────────────────────────────────────

    def test_parse_reviewed_data_valid_json_string(self):
        """Valid JSON string is parsed."""
        result = _parse_reviewed_data('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_reviewed_data_json_array(self):
        """JSON array string is parsed."""
        result = _parse_reviewed_data('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_parse_reviewed_data_json_simple_string(self):
        """JSON quoted string is parsed."""
        result = _parse_reviewed_data('"hello"')
        assert result == "hello"

    def test_parse_reviewed_data_invalid_json(self):
        """Invalid JSON string is returned as-is."""
        result = _parse_reviewed_data("not json {")
        assert result == "not json {"

    def test_parse_reviewed_data_plain_string(self):
        """Plain string (not JSON) is returned as-is."""
        result = _parse_reviewed_data("plain text")
        assert result == "plain text"

    def test_parse_reviewed_data_empty_string(self):
        """Empty string returns empty string (falsy)."""
        result = _parse_reviewed_data("")
        assert result == ""

    def test_parse_reviewed_data_none(self):
        """None returns None."""
        result = _parse_reviewed_data(None)
        assert result is None

    def test_parse_reviewed_data_dict_passthrough(self):
        """Already-parsed dict is returned as-is."""
        data = {"key": "value"}
        result = _parse_reviewed_data(data)
        assert result is data

    def test_parse_reviewed_data_list_passthrough(self):
        """Already-parsed list is returned as-is."""
        data = [1, 2, 3]
        result = _parse_reviewed_data(data)
        assert result is data
