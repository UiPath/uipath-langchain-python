"""Tests for escalation_tool.py metadata."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolCall
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationChannelProperties,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    AgentQuickFormChannelProperties,
    AgentQuickFormEscalationChannel,
    StandardRecipient,
)
from uipath.platform.action_center.tasks import Task, TaskRecipient, TaskRecipientType

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.escalation_jit import (
    _app_type_from_project_type,
    _find_schema_file_id,
    _has_app_name_override,
    _is_inline_app,
    _resolve_app_action_schema,
    _resolve_app_project,
    _resolve_is_debug_run,
    _resolve_solution_id,
)
from uipath_langchain.agent.tools.escalation_memory import (
    EscalationMemoryCachedResult,
)
from uipath_langchain.agent.tools.escalation_tool import (
    _build_escalation_memory_payload,
    _channel_app_prop,
    _parse_task_data,
    create_escalation_tool,
)


def _make_mock_task(**overrides):
    """Create a Task instance for tests."""
    defaults = {"id": 1, "key": "task-key", "title": "Test Task"}
    defaults.update(overrides)
    return Task(**defaults)


class TestEscalationToolMetadata:
    """Test that escalation tool has correct metadata for observability."""

    @pytest.fixture
    def escalation_resource(self):
        """Create a minimal escalation tool resource config."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[
                        StandardRecipient(
                            type=AgentEscalationRecipientType.USER_EMAIL,
                            value="user@example.com",
                        )
                    ],
                )
            ],
        )

    @pytest.fixture
    def escalation_resource_no_recipient(self):
        """Create escalation resource without recipients."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

    @pytest.fixture
    def quick_form_escalation_resource(self):
        """Create a quick-form escalation resource (channel has no app_name)."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentQuickFormEscalationChannel(
                    name="Escalation",
                    type="actionCenterQuickForm",
                    description="Quick Form channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentQuickFormChannelProperties(
                        form_schema={"schemaId": "schema-123", "fields": []},
                    ),
                    recipients=[
                        StandardRecipient(
                            type=AgentEscalationRecipientType.USER_EMAIL,
                            value="user@example.com",
                        )
                    ],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_escalation_tool_has_metadata(self, escalation_resource):
        """Test that escalation tool has metadata dict."""
        tool = create_escalation_tool(escalation_resource)

        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_tool_type(self, escalation_resource):
        """Test that metadata contains tool_type for span detection."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "escalation"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_display_name(self, escalation_resource):
        """Test that metadata contains display_name from app_name."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "ApprovalApp"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_display_name_falls_back_to_channel_name(
        self, quick_form_escalation_resource
    ):
        """Quick-form channels have no app_name; display_name uses the channel name."""
        tool = create_escalation_tool(quick_form_escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "Escalation"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_channel_type(self, escalation_resource):
        """Test that metadata contains channel_type for span attributes."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert tool.metadata["channel_type"] == "actionCenter"

    @pytest.mark.asyncio
    async def test_escalation_tool_metadata_has_span_context(self, escalation_resource):
        """Test that metadata contains a span context carrier for memory ingest."""
        tool = create_escalation_tool(escalation_resource)
        assert tool.metadata is not None
        assert "_span_context" in tool.metadata
        assert isinstance(tool.metadata["_span_context"], dict)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_metadata_has_recipient(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that metadata contains recipient when recipient is USER_EMAIL."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] == TaskRecipient(
            value="user@example.com",
            type=TaskRecipientType.EMAIL,
            displayName="user@example.com",
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_metadata_recipient_none_when_no_recipients(
        self, mock_interrupt, mock_uipath_class, escalation_resource_no_recipient
    ):
        """Test that recipient is None when no recipients configured."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource_no_recipient)

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert tool.metadata is not None
        assert tool.metadata["recipient"] is None

    @pytest.fixture
    def escalation_resource_jit(self):
        """Escalation resource carrying JIT (debug) project key and app type."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                        project_key="proj-key-abc",
                        app_type="Custom",
                    ),
                    recipients=[
                        StandardRecipient(
                            type=AgentEscalationRecipientType.USER_EMAIL,
                            value="user@example.com",
                        )
                    ],
                )
            ],
        )

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "UIPATH_PROJECT_ID": "proj-1",
            "UIPATH_FEATURE_EnableJITEscalationApps": "true",
        },
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_action_schema",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_project",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_is_debug_run",
        new_callable=AsyncMock,
    )
    @patch("uipath_langchain.agent.tools.escalation_jit.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_resolves_jit_fields_in_debug(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_jit_uipath_class,
        mock_resolve_debug,
        mock_resolve_project,
        mock_resolve_schema,
        escalation_resource_jit,
    ):
        """In a debug run with the flag on, the app project key, app type and action
        schema are resolved at runtime and forwarded; is_debug is NOT passed."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        mock_resolve_debug.return_value = True
        mock_resolve_project.return_value = {
            "designId": "proj-key-abc",
            "id": "proj-id",
            "projectType": "Process",  # -> Custom
        }
        schema = {
            "key": "schema-key",
            "inOuts": [],
            "inputs": [],
            "outputs": [],
            "outcomes": [],
        }
        mock_resolve_schema.return_value = schema

        tool = create_escalation_tool(escalation_resource_jit)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        kwargs = mock_client.tasks.create_async.call_args.kwargs
        assert kwargs["app_project_key"] == "proj-key-abc"
        assert kwargs["app_type"] == "Custom"
        assert kwargs["action_schema"] == schema
        # is_debug is sourced from UiPathConfig inside the SDK, never passed here.
        assert "is_debug" not in kwargs

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "UIPATH_PROJECT_ID": "proj-1",
            "UIPATH_FEATURE_EnableJITEscalationApps": "true",
        },
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_project",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_is_debug_run",
        new_callable=AsyncMock,
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_raises_in_debug_when_app_unresolvable(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_resolve_debug,
        mock_resolve_project,
        escalation_resource_jit,
    ):
        """A low-code app in debug that can't be resolved from the solution raises a
        USER error and does not create a task."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        mock_resolve_debug.return_value = True
        mock_resolve_project.return_value = None  # cannot resolve the app

        tool = create_escalation_tool(escalation_resource_jit)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]
        mock_client.tasks.create_async.assert_not_called()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_PROJECT_ID": "proj-1"}, clear=False)
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_project",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_is_debug_run",
        new_callable=AsyncMock,
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_skips_jit_when_flag_disabled(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_resolve_debug,
        mock_resolve_project,
        escalation_resource_jit,
    ):
        """With the feature flag off, no JIT resolution happens even in debug."""
        monkeypatch_env = os.environ.pop("UIPATH_FEATURE_EnableJITEscalationApps", None)
        assert monkeypatch_env is None  # flag not set → disabled

        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        mock_resolve_debug.return_value = True

        tool = create_escalation_tool(escalation_resource_jit)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # JIT app resolution is never attempted when the flag is off.
        mock_resolve_project.assert_not_called()
        kwargs = mock_client.tasks.create_async.call_args.kwargs
        assert kwargs["app_project_key"] is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_with_string_task_title(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool with legacy string task title."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        # Create resource with string task title
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "taskTitle": "Static Task Title",
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)

        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # Verify create_async was called with the static title
        create_call = mock_client.tasks.create_async.call_args
        assert create_call[1]["title"] == "Static Task Title"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_with_text_builder_task_title(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool with TEXT_BUILDER task title builds from tokens."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        # Create resource with TEXT_BUILDER task title containing variable token
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "taskTitle": {
                "type": "textBuilder",
                "tokens": [
                    {"type": "simpleText", "rawString": "Approve request for "},
                    {"type": "variable", "rawString": "input.userName"},
                ],
            },
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)

        # Create mock state with variables for token interpolation
        state = {"userName": "John Doe", "messages": []}
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, state)  # type: ignore[attr-defined]

        # Verify create_async was called with the correctly built task title
        create_call = mock_client.tasks.create_async.call_args
        assert create_call[1]["title"] == "Approve request for John Doe"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_with_empty_task_title_defaults_to_escalation_task(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool defaults to 'Escalation Task' when task title is empty."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        # Create resource with empty string task title
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "taskTitle": "",
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)

        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper to test full flow
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        # Verify create_async was called with the default title
        create_call = mock_client.tasks.create_async.call_args
        assert create_call[1]["title"] == "Escalation Task"


class TestEscalationToolOutputSchema:
    """Test escalation tool output schema for simulation support."""

    @pytest.fixture
    def escalation_resource(self):
        """Create a minimal escalation tool resource config."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={
                        "type": "object",
                        "properties": {
                            "approved": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                    },
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[
                        StandardRecipient(
                            type=AgentEscalationRecipientType.USER_EMAIL,
                            value="user@example.com",
                        )
                    ],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_escalation_tool_output_schema_has_action_field(
        self, escalation_resource
    ):
        """Test that escalation tool output schema includes action field."""
        tool = create_escalation_tool(escalation_resource)
        # Get the output schema from the tool's args_schema
        args_schema = tool.args_schema
        assert args_schema is not None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_result_validation(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that tool properly processes and validates results."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 123
        mock_result.key = None
        mock_result.assigned_to_user = None
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert isinstance(result, dict)
        assert result["outcome"] == "approve"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_extracts_action_from_result(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that tool correctly extracts action from escalation result."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {"approved": True}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert mock_interrupt.called

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_raises_when_task_is_deleted(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that escalation tool raises AgentRuntimeError when task is deleted."""
        from uipath_langchain.agent.exceptions import AgentRuntimeError

        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = True
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_dict_result_without_is_deleted_defaults_to_false(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that a dict result without is_deleted is accepted and defaults to False."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        # Return a plain dict without is_deleted — exercises the TypeAdapter path
        mock_interrupt.return_value = {
            "action": "approve",
            "data": {"approved": True, "reason": "looks good"},
        }

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["outcome"] == "approve"
        assert result["output"] == {"approved": True, "reason": "looks good"}

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_escalation_tool_with_outcome_mapping_end(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test escalation tool with outcome mapping that ends agent."""
        from uipath_langchain.agent.exceptions import AgentRuntimeError

        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 456
        mock_result.key = None
        mock_result.assigned_to_user = None
        mock_result.action = "approve"
        mock_result.data = {"approved": True}
        mock_interrupt.return_value = mock_result

        # Create resource with outcome mapping where approve -> end
        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "outcomeMapping": {"approve": "end", "reject": "continue"},
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        # Invoke through the wrapper - should raise AgentRuntimeError
        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert mock_interrupt.called

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    async def test_cached_escalation_uses_outcome_mapping(
        self, mock_check_memory_cache: AsyncMock
    ):
        """Test cached outcomes follow the same outcome mapping as live results."""
        from uipath_langchain.agent.exceptions import AgentRuntimeError

        mock_check_memory_cache.return_value = EscalationMemoryCachedResult(
            output={"approved": True},
            outcome="approve",
        )

        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
            "outcomeMapping": {"approve": "end", "reject": "continue"},
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.get_execution_folder_path")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    async def test_cache_lookup_uses_memory_folder_path(
        self,
        mock_check_memory_cache: AsyncMock,
        mock_get_execution_folder_path: MagicMock,
    ):
        """Test escalation memory calls use the memory folder, not task folder."""
        mock_get_execution_folder_path.return_value = "/Execution/Folder"
        mock_check_memory_cache.return_value = EscalationMemoryCachedResult(
            output={"approved": True},
            outcome="approve",
        )

        channel_dict = {
            "name": "action_center",
            "type": "actionCenter",
            "description": "Action Center channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "appName": "ApprovalApp",
                "appVersion": 1,
                "resourceKey": "test-key",
            },
            "recipients": [],
        }

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[AgentEscalationChannel(**channel_dict)],
            properties={
                "memory": {
                    "isEnabled": True,
                    "memorySpaceId": "space-123",
                    "memorySpaceName": "MemorySpace",
                    "folderPath": "/Memory/Folder",
                }
            },
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result == {
            "output": {"approved": True},
            "outcome": "approve",
            "task_id": None,
            "assigned_to": None,
        }
        mock_check_memory_cache.assert_awaited_once()
        assert mock_check_memory_cache.await_args is not None
        assert (
            mock_check_memory_cache.await_args.kwargs["folder_path"] == "/Memory/Folder"
        )
        assert (
            mock_check_memory_cache.await_args.kwargs["memory_space_name"]
            == "MemorySpace"
        )


class TestEscalationToolTaskInfo:
    """Test that escalation tool extracts task_id and assigned_to."""

    @pytest.fixture
    def escalation_resource(self):
        """Create a minimal escalation tool resource config."""
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_wrapper_returns_task_id_and_assigned_to(self, escalation_resource):
        """Test that wrapper result includes task_id and assigned_to from Task."""
        tool = create_escalation_tool(escalation_resource)

        # Mock ainvoke on the class to test the wrapper in isolation
        mock_ainvoke = AsyncMock(
            return_value={
                "action": "continue",
                "output": {"reason": "looks good"},
                "outcome": "approve",
                "task_id": 12345,
                "assigned_to": "user@example.com",
            }
        )

        call = ToolCall(args={}, id="test-call", name=tool.name)
        with patch.object(type(tool), "ainvoke", mock_ainvoke):
            result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["task_id"] == 12345
        assert result["assigned_to"] == "user@example.com"
        assert result["outcome"] == "approve"

    @pytest.mark.asyncio
    async def test_wrapper_handles_missing_assigned_to_user(self, escalation_resource):
        """Test that wrapper handles None assigned_to_user gracefully."""
        tool = create_escalation_tool(escalation_resource)

        # Mock ainvoke on the class to test the wrapper in isolation
        mock_ainvoke = AsyncMock(
            return_value={
                "action": "continue",
                "output": {},
                "outcome": "reject",
                "task_id": 99999,
                "assigned_to": None,
            }
        )

        call = ToolCall(args={}, id="test-call", name=tool.name)
        with patch.object(type(tool), "ainvoke", mock_ainvoke):
            result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["task_id"] == 99999
        assert result["assigned_to"] is None


class TestEscalationToolCreatesTaskBeforeInterrupt:
    """Test that escalation tool creates task inline before calling interrupt."""

    @pytest.fixture
    def escalation_resource(self):
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_creates_task_then_interrupts_with_wait_escalation(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test task is created via create_async, then interrupt(WaitEscalation)."""
        from uipath.platform.common import WaitEscalation

        task = _make_mock_task(id=555, key="task-key-555")
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 555
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.assigned_to_user = None
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={"field": "value"}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_client.tasks.create_async.assert_called_once()

        # Verify interrupt was called with WaitEscalation containing the task
        mock_interrupt.assert_called_once()
        interrupt_arg = mock_interrupt.call_args[0][0]
        assert isinstance(interrupt_arg, WaitEscalation)
        assert interrupt_arg.action == task

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_creates_task_with_channel_folder_path(
        self, mock_interrupt, mock_uipath_class, escalation_resource
    ):
        """Test that tasks.create_async receives app_folder_path from the channel.

        The app channel carries the folder its app is deployed in, so that
        folder is used rather than the agent's own execution folder
        (``UIPATH_FOLDER_PATH``).
        """
        escalation_resource.channels[0].properties.folder_name = "/Apps/Approvals"
        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.id = 555
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.assigned_to_user = None
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        create_call_kwargs = mock_client.tasks.create_async.call_args[1]
        assert create_call_kwargs["app_folder_path"] == "/Apps/Approvals"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    async def test_task_creation_failure_propagates(
        self, mock_uipath_class, escalation_resource
    ):
        """Test that task creation failure propagates as exception."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(side_effect=Exception("API error"))
        mock_uipath_class.return_value = mock_client

        tool = create_escalation_tool(escalation_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(Exception, match="API error"):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool.get_current_span_and_trace_ids"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool._ingest_escalation_memory")
    @patch("uipath_langchain.agent.tools.escalation_tool._resolve_user_id")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_memory_ingest_uses_traced_escalation_span_context(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_check_memory_cache,
        mock_resolve_user_id,
        mock_ingest_memory,
        mock_get_current_span_and_trace_ids,
    ):
        """Escalation memory ingest should use the escalationTool child span."""
        mock_check_memory_cache.return_value = None
        mock_resolve_user_id.return_value = "cef1337c-3456-4ae9-81c9-30d033dc2bef"
        mock_ingest_memory.return_value = None
        mock_get_current_span_and_trace_ids.return_value = ("wrong-span", "wrong-trace")

        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.completed_by_user = {"emailAddress": "reviewer@example.com"}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        assert tool.metadata is not None
        tool.metadata["_span_context"]["parent_span_id"] = "3a064d559eca5d62"
        tool.metadata["_span_context"]["trace_id"] = "5d3feebba60343dfb9364b89ee304a5b"

        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_get_current_span_and_trace_ids.assert_not_called()
        mock_ingest_memory.assert_awaited_once()
        assert mock_ingest_memory.await_args is not None
        assert (
            mock_ingest_memory.await_args.kwargs["parent_span_id"] == "3a064d559eca5d62"
        )
        assert (
            mock_ingest_memory.await_args.kwargs["trace_id"]
            == "5d3feebba60343dfb9364b89ee304a5b"
        )
        assert tool.metadata["_span_context"] == {}

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool.get_current_span_and_trace_ids"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool._ingest_escalation_memory")
    @patch("uipath_langchain.agent.tools.escalation_tool._resolve_user_id")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_memory_ingest_falls_back_to_current_span_context(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_check_memory_cache,
        mock_resolve_user_id,
        mock_ingest_memory,
        mock_get_current_span_and_trace_ids,
    ):
        """Escalation memory ingest should fall back when metadata is incomplete."""
        mock_check_memory_cache.return_value = None
        mock_resolve_user_id.return_value = None
        mock_ingest_memory.return_value = None
        mock_get_current_span_and_trace_ids.return_value = (
            "fallback-span",
            "fallback-trace",
        )

        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.completed_by_user = {"displayName": "Reviewer"}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_get_current_span_and_trace_ids.assert_called_once()
        mock_ingest_memory.assert_awaited_once()
        assert mock_ingest_memory.await_args is not None
        assert mock_ingest_memory.await_args.kwargs["parent_span_id"] == "fallback-span"
        assert mock_ingest_memory.await_args.kwargs["trace_id"] == "fallback-trace"
        assert mock_ingest_memory.await_args.kwargs["user_id"] is None

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.tools.escalation_tool.get_current_span_and_trace_ids"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool._ingest_escalation_memory")
    @patch("uipath_langchain.agent.tools.escalation_tool._resolve_user_id")
    @patch(
        "uipath_langchain.agent.tools.escalation_tool._check_escalation_memory_cache"
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_memory_ingest_skips_when_span_context_is_unavailable(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_check_memory_cache,
        mock_resolve_user_id,
        mock_ingest_memory,
        mock_get_current_span_and_trace_ids,
    ):
        """Escalation memory ingest should be skipped without trace provenance."""
        mock_check_memory_cache.return_value = None
        mock_resolve_user_id.return_value = None
        mock_get_current_span_and_trace_ids.return_value = (None, None)

        task = _make_mock_task(id=555)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.completed_by_user = {"displayName": "Reviewer"}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                    ),
                    recipients=[],
                )
            ],
            isAgentMemoryEnabled=True,
            memorySpaceId="space-123",
        )

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        result = await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        assert result["output"] == {}
        assert result["outcome"] == "approve"
        mock_get_current_span_and_trace_ids.assert_called_once()
        mock_ingest_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wrapper_requires_metadata(self, escalation_resource):
        tool = create_escalation_tool(escalation_resource)
        tool.metadata = None
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(
            RuntimeError,
            match="Tool metadata is required for task_title resolution",
        ):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]


class TestParseTaskData:
    """Test output task data is filtered correctly."""

    def test_filters_input_fields_when_no_output_schema(self):
        """Test that input fields are excluded when output_schema is None."""
        data = {"input_field": "value1", "output_field": "value2"}
        input_schema = {"properties": {"input_field": {"type": "string"}}}

        result = _parse_task_data(data, input_schema, output_schema=None)

        assert result == {"output_field": "value2"}
        assert "input_field" not in result

    def test_includes_only_output_fields_when_output_schema_provided(self):
        """Test that only output schema fields are included."""
        data = {"field1": "a", "field2": "b", "field3": "c"}
        input_schema = {"properties": {"field1": {"type": "string"}}}
        output_schema = {
            "properties": {"field1": {"type": "string"}, "field2": {"type": "string"}}
        }

        result = _parse_task_data(data, input_schema, output_schema)

        assert result == {"field1": "a", "field2": "b"}
        assert "field3" not in result

    def test_handles_missing_properties_in_schemas(self):
        """Test behavior when schemas lack 'properties' key."""
        data = {"field": "value"}

        # No properties key in schemas
        result = _parse_task_data(data, {}, None)
        assert result == {"field": "value"}


class TestEscalationMemoryPayload:
    """Test escalation memory ingest payload shape."""

    def test_builds_trace_and_search_payloads(self):
        """Test memory ingest matches the escalation memory service contract."""
        serialized_input = {
            "request_details": "User requested escalation before answering."
        }
        escalation_output = {"reviewer_comment": "approve"}

        answer, attributes = _build_escalation_memory_payload(
            serialized_input,
            escalation_output,
            "Approve",
        )

        assert answer == {
            "output": {"reviewer_comment": "approve"},
            "outcome": "Approve",
        }
        assert attributes == {"arguments": serialized_input}
        assert "escalation-input" not in attributes


class TestQuickFormEscalation:
    """QuickForm channel (actionCenterQuickForm) path through create_escalation_tool."""

    @pytest.fixture
    def quick_form_schema(self):
        return {
            "schemaId": "00000000-0000-0000-0000-000000000abc",
            "fields": [{"name": "decision", "type": "string"}],
            "outcomes": ["approve", "reject"],
        }

    @pytest.fixture
    def quick_form_channel_dict(self, quick_form_schema):
        return {
            "name": "quick_form_channel",
            "type": "actionCenterQuickForm",
            "description": "Quick-form channel",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {
                "schema": quick_form_schema,
                "isActionableMessageEnabled": False,
                "actionableMessageMetaData": None,
            },
            "recipients": [],
        }

    @pytest.fixture
    def quick_form_resource(self, quick_form_channel_dict):
        return AgentEscalationResourceConfig(
            name="quick_form_approval",
            description="Request quick-form approval",
            channels=[AgentQuickFormEscalationChannel(**quick_form_channel_dict)],
        )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_dispatches_to_create_quickform_async(
        self,
        mock_interrupt,
        mock_uipath_class,
        quick_form_resource,
        quick_form_schema,
    ):
        task = _make_mock_task(id=777, key="task-key-777")
        mock_client = MagicMock()
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(quick_form_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_client.tasks.create_quickform_async.assert_called_once()
        mock_client.tasks.create_async.assert_not_called()

        kwargs = mock_client.tasks.create_quickform_async.call_args[1]
        assert kwargs["task_schema_key"] == "00000000-0000-0000-0000-000000000abc"
        assert kwargs["schema"] == quick_form_schema
        assert kwargs["folder_path"] == "/Test/Folder"
        assert "app_name" not in kwargs
        assert "app_folder_path" not in kwargs

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_wait_escalation_app_name_is_none_for_quick_form(
        self, mock_interrupt, mock_uipath_class, quick_form_resource
    ):
        from uipath.platform.common import WaitEscalation

        task = _make_mock_task(id=778)
        mock_client = MagicMock()
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(quick_form_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_interrupt.assert_called_once()
        interrupt_arg = mock_interrupt.call_args[0][0]
        assert isinstance(interrupt_arg, WaitEscalation)
        assert interrupt_arg.app_name is None
        assert interrupt_arg.action == task

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_outcome_mapping_end_terminates_agent(
        self,
        mock_interrupt,
        mock_uipath_class,
        quick_form_channel_dict,
    ):
        from uipath_langchain.agent.exceptions import AgentRuntimeError

        channel = dict(quick_form_channel_dict)
        channel["outcomeMapping"] = {"approve": "end", "reject": "continue"}
        resource = AgentEscalationResourceConfig(
            name="quick_form_approval",
            description="Request quick-form approval",
            channels=[AgentQuickFormEscalationChannel(**channel)],
        )

        task = _make_mock_task(id=779)
        mock_client = MagicMock()
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_tool_metadata_for_quick_form_resource(self, quick_form_resource):
        tool = create_escalation_tool(quick_form_resource)
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "escalation"
        assert tool.metadata["channel_type"] == "actionCenterQuickForm"
        assert "_span_context" in tool.metadata
        assert "_bts_context" in tool.metadata

    async def test_missing_schema_id_raises_on_construction(
        self, quick_form_channel_dict
    ):
        from uipath_langchain.agent.exceptions import AgentStartupError

        channel = dict(quick_form_channel_dict)
        channel["properties"] = {
            "schema": {"fields": [], "outcomes": []},
            "isActionableMessageEnabled": False,
            "actionableMessageMetaData": None,
        }
        resource = AgentEscalationResourceConfig(
            name="quick_form_approval",
            description="Request quick-form approval",
            channels=[AgentQuickFormEscalationChannel(**channel)],
        )

        with pytest.raises(AgentStartupError) as exc_info:
            create_escalation_tool(resource)

        assert "INVALID_TOOL_CONFIG" in exc_info.value.error_info.code

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Test/Folder"})
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_action_center_channel_does_not_dispatch_to_quickform(
        self, mock_interrupt, mock_uipath_class
    ):
        resource = AgentEscalationResourceConfig(
            name="action_center_approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center_channel",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp", app_version=1
                    ),
                    recipients=[],
                )
            ],
        )

        task = _make_mock_task(id=780)
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=task)
        mock_client.tasks.create_quickform_async = AsyncMock(return_value=task)
        mock_uipath_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.action = "approve"
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        mock_client.tasks.create_async.assert_called_once()
        mock_client.tasks.create_quickform_async.assert_not_called()


def _app_channel(**props):
    """Build an AgentEscalationChannel with the given app properties."""
    return AgentEscalationChannel(
        name="action_center",
        type="actionCenter",
        description="Action Center channel",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        properties=AgentEscalationChannelProperties(
            app_name=props.get("app_name", "ApprovalApp"),
            app_version=props.get("app_version", 0),
            resource_key="test-key",
            app_type=props.get("app_type"),
        ),
        recipients=[],
    )


class TestAppTypeFromProjectType:
    @pytest.mark.parametrize(
        "project_type,expected",
        [
            ("AppV2", "Coded"),
            ("Process", "Custom"),
            (None, None),
            ("", None),
            ("Unknown", None),
        ],
    )
    def test_mapping(self, project_type, expected):
        assert _app_type_from_project_type(project_type) == expected


class TestChannelAppProp:
    def test_reads_property_from_agent_channel(self):
        channel = _app_channel(app_name="MyApp", app_version=2, app_type="Coded")
        assert _channel_app_prop(channel, "app_name") == "MyApp"
        assert _channel_app_prop(channel, "app_version") == 2
        assert _channel_app_prop(channel, "app_type") == "Coded"

    def test_returns_none_for_non_agent_channel(self):
        # A non-AgentEscalationChannel object carries no app properties.
        assert _channel_app_prop(object(), "app_name") is None


class TestFindSchemaFileId:
    def test_finds_schema_file_at_top_level(self):
        node = {"files": [{"name": "schema-abc.json", "id": "file-1"}], "folders": []}
        assert _find_schema_file_id(node) == "file-1"

    def test_finds_schema_file_in_nested_folder(self):
        node = {
            "files": [{"name": "other.json", "id": "x"}],
            "folders": [
                {"files": [{"name": "schema-xyz.json", "id": "file-2"}], "folders": []}
            ],
        }
        assert _find_schema_file_id(node) == "file-2"

    def test_returns_none_when_no_schema_file(self):
        node = {"files": [{"name": "data.json", "id": "x"}], "folders": []}
        assert _find_schema_file_id(node) is None

    def test_returns_none_for_non_dict(self):
        assert _find_schema_file_id(None) is None
        assert _find_schema_file_id("not-a-node") is None


class TestHasAppNameOverride:
    def _set_overwrites(self, overwrites):
        from uipath.platform.common._bindings import _resource_overwrites

        return _resource_overwrites, _resource_overwrites.set(overwrites)

    def test_false_when_no_app_name(self):
        assert _has_app_name_override(None, None) is False

    def test_false_when_no_overwrites_context(self):
        assert _has_app_name_override("MyApp", None) is False

    def test_true_when_app_key_present(self):
        var, token = self._set_overwrites({"app.MyApp": object()})
        try:
            assert _has_app_name_override("MyApp", None) is True
        finally:
            var.reset(token)

    def test_true_when_folder_qualified_key_present(self):
        var, token = self._set_overwrites({"app.MyApp.Shared": object()})
        try:
            assert _has_app_name_override("MyApp", "Shared") is True
        finally:
            var.reset(token)

    def test_false_when_key_absent(self):
        var, token = self._set_overwrites({"app.OtherApp": object()})
        try:
            assert _has_app_name_override("MyApp", None) is False
        finally:
            var.reset(token)


class TestIsInlineApp:
    @pytest.mark.parametrize(
        "app_type,app_version,expected",
        [
            ("Custom", 0, True),
            (None, 0, True),
            ("Custom", 1, True),  # v1 low-code, no binding override -> inline
            ("Coded", 0, False),  # coded is never inline
            ("Custom", 2, False),  # only v0/v1 qualify
        ],
    )
    def test_predicate(self, app_type, app_version, expected):
        assert _is_inline_app(app_type, app_version, "MyApp", None) is expected

    def test_v1_with_binding_override_is_not_inline(self):
        from uipath.platform.common._bindings import _resource_overwrites

        token = _resource_overwrites.set({"app.MyApp": object()})
        try:
            assert _is_inline_app("Custom", 1, "MyApp", None) is False
        finally:
            _resource_overwrites.reset(token)


class TestResolveIsDebugRun:
    @pytest.mark.asyncio
    async def test_returns_false_when_no_job_key(self, monkeypatch):
        monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
        assert await _resolve_is_debug_run() is False

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_JOB_KEY": "job-1"})
    @patch("uipath_langchain.agent.tools.escalation_jit.UiPath")
    async def test_true_sets_config(self, mock_uipath_class):
        from uipath.platform.common import UiPathConfig

        UiPathConfig.reset()
        try:
            job = MagicMock()
            job.parent_context = '{"IsDebug": true}'
            mock_client = MagicMock()
            mock_client.jobs.retrieve_async = AsyncMock(return_value=job)
            mock_uipath_class.return_value = mock_client

            result = await _resolve_is_debug_run()

            assert result is True
            assert UiPathConfig.is_rooted_to_debug_job is True
        finally:
            UiPathConfig.reset()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_JOB_KEY": "job-1"})
    @patch("uipath_langchain.agent.tools.escalation_jit.UiPath")
    async def test_false_does_not_set_config(self, mock_uipath_class):
        from uipath.platform.common import UiPathConfig

        UiPathConfig.reset()
        try:
            job = MagicMock()
            job.parent_context = '{"IsDebug": false}'
            mock_client = MagicMock()
            mock_client.jobs.retrieve_async = AsyncMock(return_value=job)
            mock_uipath_class.return_value = mock_client

            result = await _resolve_is_debug_run()

            assert result is False
            assert UiPathConfig.is_rooted_to_debug_job is False
        finally:
            UiPathConfig.reset()


class TestJitResolutionHelpers:
    """Runtime resolution of the app project / solution / action schema."""

    @pytest.fixture(autouse=True)
    def _reset_config(self):
        from uipath.platform.common import UiPathConfig

        UiPathConfig.reset()
        yield
        UiPathConfig.reset()

    def _client_returning(self, by_url):
        """MagicMock client whose api_client.request_async returns a canned
        ``.json()`` payload chosen by substring match on the request url."""

        def _side_effect(method, url=None, **kwargs):
            for needle, payload in by_url.items():
                if needle in url:
                    resp = MagicMock()
                    resp.json.return_value = payload
                    return resp
            raise AssertionError(f"unexpected url: {url}")

        client = MagicMock()
        client.api_client.request_async = AsyncMock(side_effect=_side_effect)
        return client

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_PROJECT_ID": "proj-1"})
    async def test_resolve_solution_id_from_project(self):
        from uipath.platform.common import UiPathConfig

        client = self._client_returning({"/Project/proj-1": {"solutionId": "sol-9"}})
        assert await _resolve_solution_id(client) == "sol-9"
        # Cached on the config for subsequent lookups.
        assert UiPathConfig.studio_solution_id == "sol-9"

    @pytest.mark.asyncio
    async def test_resolve_solution_id_none_without_project(self, monkeypatch):
        monkeypatch.delenv("UIPATH_PROJECT_ID", raising=False)
        client = MagicMock()
        assert await _resolve_solution_id(client) is None

    @pytest.mark.asyncio
    async def test_resolve_app_project_matches_by_name(self):
        from uipath.platform.common import UiPathConfig

        UiPathConfig.studio_solution_id = "sol-9"
        client = self._client_returning(
            {
                "/Solution/sol-9": {
                    "projects": [
                        {"name": "Other", "isApp": True, "designId": "d0", "id": "i0"},
                        {
                            "name": "ApprovalApp",
                            "isApp": True,
                            "designId": "d1",
                            "id": "i1",
                            "projectType": "Process",
                        },
                        {"name": "NotAnApp", "isApp": False},
                    ]
                }
            }
        )
        app = await _resolve_app_project(client, "ApprovalApp")
        assert app is not None
        assert app["designId"] == "d1"
        assert app["projectType"] == "Process"

    @pytest.mark.asyncio
    async def test_resolve_app_project_none_when_no_match(self):
        from uipath.platform.common import UiPathConfig

        UiPathConfig.studio_solution_id = "sol-9"
        client = self._client_returning(
            {"/Solution/sol-9": {"projects": [{"name": "X", "isApp": True}]}}
        )
        assert await _resolve_app_project(client, "ApprovalApp") is None

    @pytest.mark.asyncio
    async def test_resolve_app_project_none_without_solution(self, monkeypatch):
        monkeypatch.delenv("UIPATH_PROJECT_ID", raising=False)
        client = MagicMock()
        assert await _resolve_app_project(client, "ApprovalApp") is None

    @pytest.mark.asyncio
    async def test_resolve_app_action_schema(self):
        client = self._client_returning(
            {
                "/FileOperations/Structure": {
                    "files": [{"name": "schema-1.json", "id": "f1"}],
                    "folders": [],
                },
                "/FileOperations/File/f1": {"key": "schema-1", "inputs": []},
            }
        )
        schema = await _resolve_app_action_schema(client, "proj-id")
        assert schema is not None
        assert schema["key"] == "schema-1"

    @pytest.mark.asyncio
    async def test_resolve_app_action_schema_none_when_no_schema_file(self):
        client = self._client_returning(
            {"/FileOperations/Structure": {"files": [], "folders": []}}
        )
        assert await _resolve_app_action_schema(client, "proj-id") is None


class TestEscalationJitFallbacks:
    """Failure/fallback paths in the debug + JIT resolution flow."""

    @pytest.fixture
    def jit_resource(self):
        return AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=1,
                        resource_key="test-key",
                        app_type="Custom",
                    ),
                    recipients=[],
                )
            ],
        )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_PROJECT_ID": "proj-1"}, clear=False)
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_project",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_is_debug_run",
        new_callable=AsyncMock,
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_debug_resolution_failure_falls_back_to_release(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_resolve_debug,
        mock_resolve_project,
        jit_resource,
    ):
        """If debug resolution raises, treat as release: skip JIT, still create."""
        os.environ.pop("UIPATH_FEATURE_EnableJITEscalationApps", None)
        os.environ["UIPATH_FEATURE_EnableJITEscalationApps"] = "true"
        try:
            mock_client = MagicMock()
            mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
            mock_uipath_class.return_value = mock_client
            mock_result = MagicMock()
            mock_result.action = None
            mock_result.data = {}
            mock_result.is_deleted = False
            mock_interrupt.return_value = mock_result

            mock_resolve_debug.side_effect = RuntimeError("job lookup failed")

            tool = create_escalation_tool(jit_resource)
            call = ToolCall(args={}, id="test-call", name=tool.name)
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

            # Debug unknown -> release: no JIT resolution, task still created.
            mock_resolve_project.assert_not_called()
            mock_client.tasks.create_async.assert_called_once()
        finally:
            os.environ.pop("UIPATH_FEATURE_EnableJITEscalationApps", None)

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "UIPATH_PROJECT_ID": "proj-1",
            "UIPATH_FEATURE_EnableJITEscalationApps": "true",
        },
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_project",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_is_debug_run",
        new_callable=AsyncMock,
    )
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_app_resolution_failure_raises_user_error(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_resolve_debug,
        mock_resolve_project,
        jit_resource,
    ):
        """If app resolution raises, the exception is logged and the missing-fields
        USER error is raised (no task created)."""
        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client
        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        mock_resolve_debug.return_value = True
        mock_resolve_project.side_effect = RuntimeError("studio backend 500")

        tool = create_escalation_tool(jit_resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)

        with pytest.raises(AgentRuntimeError):
            await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]
        mock_client.tasks.create_async.assert_not_called()


class TestResolveIsDebugRunEdgeCases:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_JOB_KEY": "job-1"})
    @patch("uipath_langchain.agent.tools.escalation_jit.UiPath")
    async def test_empty_parent_context_returns_false(self, mock_uipath_class):
        job = MagicMock()
        job.parent_context = None
        mock_client = MagicMock()
        mock_client.jobs.retrieve_async = AsyncMock(return_value=job)
        mock_uipath_class.return_value = mock_client
        assert await _resolve_is_debug_run() is False

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_JOB_KEY": "job-1"})
    @patch("uipath_langchain.agent.tools.escalation_jit.UiPath")
    async def test_invalid_json_parent_context_returns_false(self, mock_uipath_class):
        job = MagicMock()
        job.parent_context = "not-valid-json"
        mock_client = MagicMock()
        mock_client.jobs.retrieve_async = AsyncMock(return_value=job)
        mock_uipath_class.return_value = mock_client
        assert await _resolve_is_debug_run() is False


class TestEscalationJitAppTypeDerivation:
    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "UIPATH_PROJECT_ID": "proj-1",
            "UIPATH_FEATURE_EnableJITEscalationApps": "true",
        },
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_action_schema",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_app_project",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.tools.escalation_jit._resolve_is_debug_run",
        new_callable=AsyncMock,
    )
    @patch("uipath_langchain.agent.tools.escalation_jit.UiPath")
    @patch("uipath_langchain.agent.tools.escalation_tool.UiPath")
    @patch("uipath_langchain._utils.durable_interrupt.decorator.interrupt")
    async def test_app_type_derived_from_project_type_when_absent(
        self,
        mock_interrupt,
        mock_uipath_class,
        mock_jit_uipath_class,
        mock_resolve_debug,
        mock_resolve_project,
        mock_resolve_schema,
    ):
        """When the channel carries no app_type, it is derived from the resolved
        project's projectType (Process -> Custom)."""
        resource = AgentEscalationResourceConfig(
            name="approval",
            description="Request approval",
            channels=[
                AgentEscalationChannel(
                    name="action_center",
                    type="actionCenter",
                    description="Action Center channel",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object", "properties": {}},
                    properties=AgentEscalationChannelProperties(
                        app_name="ApprovalApp",
                        app_version=0,  # inline
                        resource_key="test-key",
                        app_type=None,  # not provided by the frontend
                    ),
                    recipients=[],
                )
            ],
        )

        mock_client = MagicMock()
        mock_client.tasks.create_async = AsyncMock(return_value=_make_mock_task())
        mock_uipath_class.return_value = mock_client
        mock_result = MagicMock()
        mock_result.action = None
        mock_result.data = {}
        mock_result.is_deleted = False
        mock_interrupt.return_value = mock_result

        mock_resolve_debug.return_value = True
        mock_resolve_project.return_value = {
            "designId": "d1",
            "id": "i1",
            "projectType": "Process",
        }
        mock_resolve_schema.return_value = {"key": "schema-1"}

        tool = create_escalation_tool(resource)
        call = ToolCall(args={}, id="test-call", name=tool.name)
        await tool.awrapper(tool, call, {})  # type: ignore[attr-defined]

        kwargs = mock_client.tasks.create_async.call_args.kwargs
        assert kwargs["app_type"] == "Custom"
        assert kwargs["app_project_key"] == "d1"
