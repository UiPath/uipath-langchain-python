"""End-to-end trace parity tests for C# SDK compatibility.

These tests verify that the Python SDK trace format matches the C# SDK
temporal trace format for all span types.
"""

from uipath_agents._observability.llmops.spans.span_attributes.agent import (
    AgentOutputSpanAttributes,
    AgentRunSpanAttributes,
)
from uipath_agents._observability.llmops.spans.span_attributes.llm import (
    CompletionSpanAttributes,
    LlmCallSpanAttributes,
    ModelSettings,
    ToolCall,
    Usage,
)
from uipath_agents._observability.llmops.spans.span_attributes.tool_call import (
    ToolCallSpanAttributes,
)
from uipath_agents._observability.llmops.spans.span_attributes.tools import (
    ActionCenterToolSpanAttributes,
    AgenticProcessToolSpanAttributes,
    AgentToolSpanAttributes,
    ApiWorkflowToolSpanAttributes,
    EscalationToolSpanAttributes,
    IntegrationToolSpanAttributes,
    ProcessToolSpanAttributes,
)
from uipath_agents._observability.llmops.spans.span_attributes.types import SpanType


class TestSpanTypeCoverage:
    """Tests that all span types are correctly defined and accessible."""

    def test_core_span_types_exist(self) -> None:
        """Core span types should be defined in SpanType."""
        assert SpanType.AGENT_RUN == "agentRun"
        assert SpanType.COMPLETION == "completion"
        assert SpanType.LLM_CALL == "llmCall"
        assert SpanType.TOOL_CALL == "toolCall"
        assert SpanType.AGENT_OUTPUT == "agentOutput"
        assert SpanType.AGENT_INPUT == "agentInput"

    def test_tool_span_types_exist(self) -> None:
        """Tool-specific span types should be defined in SpanType."""
        assert SpanType.PROCESS_TOOL == "processTool"
        assert SpanType.AGENT_TOOL == "agentTool"
        assert SpanType.API_WORKFLOW_TOOL == "apiWorkflowTool"
        assert SpanType.AGENTIC_PROCESS_TOOL == "agenticProcessTool"
        assert SpanType.INTEGRATION_TOOL == "integrationTool"
        assert SpanType.ESCALATION_TOOL == "escalationTool"
        assert SpanType.ACTION_CENTER_TOOL == "actionCenterTool"


class TestSpanTypeField:
    """Tests that span type field values match C# SDK format."""

    def test_llm_call_type_is_llm_call(self) -> None:
        """LlmCallSpanAttributes.type is 'llmCall'."""
        attrs = LlmCallSpanAttributes(input="Test")
        assert attrs.type == "llmCall"
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "llmCall"

    def test_completion_type_is_completion(self) -> None:
        """CompletionSpanAttributes.type must be 'completion'."""
        attrs = CompletionSpanAttributes(model="gpt-4o")
        assert attrs.type == "completion"
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "completion"

    def test_agent_run_type(self) -> None:
        """AgentRunSpanAttributes.type must be 'agentRun'."""
        attrs = AgentRunSpanAttributes(agent_name="Test", agent_id="123")
        assert attrs.type == "agentRun"
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "agentRun"

    def test_tool_call_type(self) -> None:
        """ToolCallSpanAttributes.type defaults to 'toolCall'."""
        attrs = ToolCallSpanAttributes(tool_name="test")
        assert attrs.type == "toolCall"
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "toolCall"

    def test_tool_call_with_custom_type(self) -> None:
        """ToolCallSpanAttributes can have a custom span_type."""
        attrs = ToolCallSpanAttributes(tool_name="test", span_type="processTool")
        assert attrs.type == "processTool"
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "processTool"

    def test_process_tool_type(self) -> None:
        """ProcessToolSpanAttributes.type must be 'processTool'."""
        attrs = ProcessToolSpanAttributes(arguments={})
        assert attrs.type == "processTool"
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "processTool"

    def test_agent_output_type(self) -> None:
        """AgentOutputSpanAttributes.type must be 'agentOutput'."""
        attrs = AgentOutputSpanAttributes(output="test")
        assert attrs.type == "agentOutput"
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "agentOutput"


class TestFieldNamingConventions:
    """Tests that field names follow C# SDK camelCase conventions."""

    def test_agent_run_field_names(self) -> None:
        """AgentRunSpanAttributes fields should use camelCase aliases."""
        attrs = AgentRunSpanAttributes(
            agent_name="Test",
            agent_id="123",
            system_prompt="System",
            user_prompt="User",
            is_conversational=True,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        otel = attrs.to_otel_attributes()

        # Verify camelCase field names
        assert "agentName" in otel
        assert "agentId" in otel
        assert "systemPrompt" in otel
        assert "userPrompt" in otel
        assert "isConversational" in otel
        assert "inputSchema" in otel
        assert "outputSchema" in otel

        # Verify snake_case NOT present
        assert "agent_name" not in otel
        assert "agent_id" not in otel
        assert "system_prompt" not in otel
        assert "user_prompt" not in otel
        assert "is_conversational" not in otel
        assert "input_schema" not in otel
        assert "output_schema" not in otel

    def test_llm_call_field_names(self) -> None:
        """LlmCallSpanAttributes fields should use camelCase aliases."""
        attrs = LlmCallSpanAttributes(
            input="Test",
            content="Response",
            license_ref_id="license-123",
        )
        otel = attrs.to_otel_attributes()

        assert "licenseRefId" in otel
        assert "license_ref_id" not in otel

    def test_completion_usage_field_names(self) -> None:
        """Usage nested object fields should use camelCase."""
        attrs = CompletionSpanAttributes(
            model="gpt-4o",
            usage=Usage(
                completion_tokens=100,
                prompt_tokens=50,
                total_tokens=150,
                is_byo_execution=True,
                execution_deployment_type="cloud",
                is_pii_masked=False,
                llm_calls=1,
            ),
        )
        otel = attrs.to_otel_attributes()
        usage = otel.get("usage")

        assert usage is not None
        assert "completionTokens" in usage
        assert "promptTokens" in usage
        assert "totalTokens" in usage
        assert "isByoExecution" in usage
        assert "executionDeploymentType" in usage
        assert "isPiiMasked" in usage
        assert "llmCalls" in usage

    def test_tool_call_field_names(self) -> None:
        """ToolCallSpanAttributes fields should use camelCase aliases."""
        attrs = ToolCallSpanAttributes(
            tool_name="test_tool",
            call_id="call-123",
            tool_type="processTool",
        )
        otel = attrs.to_otel_attributes()

        assert "toolName" in otel
        assert "call_id" in otel
        assert "toolType" in otel
        assert "tool_name" not in otel
        assert "tool_type" not in otel

    def test_process_tool_field_names(self) -> None:
        """ProcessToolSpanAttributes fields should use camelCase aliases."""
        attrs = ProcessToolSpanAttributes(
            arguments={"log": "test"},
            job_id="job-123",
            job_details_uri="https://example.com/jobs/123",
        )
        otel = attrs.to_otel_attributes()

        assert "jobId" in otel
        assert "jobDetailsUri" in otel
        assert "job_id" not in otel
        assert "job_details_uri" not in otel


class TestModelSettingsTemperatureSerialization:
    """Tests that ModelSettings temperature follows C# serialization rules."""

    def test_temperature_zero_serializes_as_int(self) -> None:
        """Temperature 0.0 should serialize as 0 (int) for C# parity."""
        settings = ModelSettings(temperature=0.0, max_tokens=16384)
        dumped = settings.model_dump(by_alias=True)

        assert dumped["temperature"] == 0
        assert isinstance(dumped["temperature"], int)

    def test_temperature_one_serializes_as_int(self) -> None:
        """Temperature 1.0 should serialize as 1 (int)."""
        settings = ModelSettings(temperature=1.0, max_tokens=16384)
        dumped = settings.model_dump(by_alias=True)

        assert dumped["temperature"] == 1
        assert isinstance(dumped["temperature"], int)

    def test_temperature_fractional_serializes_as_float(self) -> None:
        """Temperature 0.7 should serialize as float."""
        settings = ModelSettings(temperature=0.7, max_tokens=16384)
        dumped = settings.model_dump(by_alias=True)

        assert dumped["temperature"] == 0.7
        assert isinstance(dumped["temperature"], float)

    def test_temperature_small_fraction_serializes_as_float(self) -> None:
        """Temperature 0.1 should serialize as float."""
        settings = ModelSettings(temperature=0.1, max_tokens=16384)
        dumped = settings.model_dump(by_alias=True)

        assert dumped["temperature"] == 0.1
        assert isinstance(dumped["temperature"], float)

    def test_temperature_none_stays_none(self) -> None:
        """Temperature None should stay None."""
        settings = ModelSettings(max_tokens=16384)
        dumped = settings.model_dump(by_alias=True, exclude_none=True)

        assert "temperature" not in dumped

    def test_completion_span_temperature_int_in_otel(self) -> None:
        """CompletionSpanAttributes with temp 0.0 should have int in OTEL output."""
        attrs = CompletionSpanAttributes(
            model="gpt-4o",
            settings=ModelSettings(temperature=0.0, max_tokens=16384),
        )
        otel = attrs.to_otel_attributes()

        settings = otel.get("settings")
        assert settings is not None
        assert settings["temperature"] == 0
        assert isinstance(settings["temperature"], int)


class TestProcessToolNoToolNameField:
    """Tests that ProcessToolSpanAttributes does not have toolName field."""

    def test_process_tool_excludes_tool_name(self) -> None:
        """ProcessToolSpanAttributes should NOT have toolName in output."""
        attrs = ProcessToolSpanAttributes(
            arguments={"log": "test"},
            job_id="job-123",
            job_details_uri="https://example.com/jobs/123",
        )
        otel = attrs.to_otel_attributes()

        assert "toolName" not in otel
        assert "tool_name" not in otel

    def test_process_tool_has_required_fields(self) -> None:
        """ProcessToolSpanAttributes should have type, arguments, jobId, jobDetailsUri."""
        attrs = ProcessToolSpanAttributes(
            arguments={"param": "value"},
            job_id="af2c5f8e-3d41-4d17-86e9-0a2e15742ff6",
            job_details_uri="https://example.com/jobs/af2c5f8e",
        )
        otel = attrs.to_otel_attributes()

        assert otel["type"] == "processTool"
        assert otel["arguments"] == {"param": "value"}
        assert otel["jobId"] == "af2c5f8e-3d41-4d17-86e9-0a2e15742ff6"
        assert otel["jobDetailsUri"] == "https://example.com/jobs/af2c5f8e"


class TestToolCallCallIdAndArguments:
    """Tests that ToolCallSpanAttributes properly captures callId and arguments."""

    def test_tool_call_captures_call_id(self) -> None:
        """ToolCallSpanAttributes should expose call_id in OTEL output."""
        attrs = ToolCallSpanAttributes(
            tool_name="test_tool",
            call_id="toolu_bdrk_01Lj6v2gwof4MUyRAULCJ7Sw",
            arguments={"key": "value"},
        )
        otel = attrs.to_otel_attributes()

        assert otel["call_id"] == "toolu_bdrk_01Lj6v2gwof4MUyRAULCJ7Sw"

    def test_tool_call_captures_arguments(self) -> None:
        """ToolCallSpanAttributes should expose arguments as 'input' in OTEL output."""
        attrs = ToolCallSpanAttributes(
            tool_name="test_tool",
            arguments={"log": "Current date is: 2026-02-05"},
        )
        otel = attrs.to_otel_attributes()

        assert otel["input"] == {"log": "Current date is: 2026-02-05"}

    def test_tool_call_handles_none_call_id(self) -> None:
        """ToolCallSpanAttributes should handle None call_id gracefully."""
        attrs = ToolCallSpanAttributes(
            tool_name="test_tool",
            arguments={"key": "value"},
        )
        otel = attrs.to_otel_attributes()

        assert "call_id" not in otel  # None values excluded


class TestLicenseRefIdInheritance:
    """Tests that licenseRefId is properly inherited and exposed."""

    def test_llm_call_has_license_ref_id(self) -> None:
        """LlmCallSpanAttributes should expose licenseRefId from BaseSpanAttributes."""
        attrs = LlmCallSpanAttributes(
            input="Test",
            license_ref_id="bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7",
        )
        otel = attrs.to_otel_attributes()

        assert otel["licenseRefId"] == "bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7"

    def test_completion_has_license_ref_id(self) -> None:
        """CompletionSpanAttributes should expose licenseRefId."""
        attrs = CompletionSpanAttributes(
            model="gpt-4o",
            license_ref_id="bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7",
        )
        otel = attrs.to_otel_attributes()

        assert otel["licenseRefId"] == "bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7"

    def test_agent_run_has_license_ref_id(self) -> None:
        """AgentRunSpanAttributes should expose licenseRefId."""
        attrs = AgentRunSpanAttributes(
            agent_name="Test",
            agent_id="123",
            license_ref_id="bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7",
        )
        otel = attrs.to_otel_attributes()

        assert otel["licenseRefId"] == "bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7"


class TestAgentRunOutputCapture:
    """Tests that AgentRunSpanAttributes properly captures output."""

    def test_agent_run_has_output_field(self) -> None:
        """AgentRunSpanAttributes should expose output field."""
        attrs = AgentRunSpanAttributes(
            agent_name="Test",
            agent_id="123",
            output={"content": "Summary: The current date has been logged."},
        )
        otel = attrs.to_otel_attributes()

        assert otel["output"] == {
            "content": "Summary: The current date has been logged."
        }

    def test_agent_run_output_with_string(self) -> None:
        """AgentRunSpanAttributes should handle string output."""
        attrs = AgentRunSpanAttributes(
            agent_name="Test",
            agent_id="123",
            output="Plain text output",
        )
        otel = attrs.to_otel_attributes()

        assert otel["output"] == "Plain text output"


class TestAdditionalToolSpanTypes:
    """Tests for additional tool span type attributes."""

    def test_agent_tool_has_job_fields(self) -> None:
        """AgentToolSpanAttributes should have jobId and jobDetailsUri."""
        attrs = AgentToolSpanAttributes(
            tool_name="sub_agent",
            arguments={"task": "research"},
            job_id="job-456",
            job_details_uri="https://example.com/jobs/456",
        )
        otel = attrs.to_otel_attributes()

        assert otel["type"] == "agentTool"
        assert otel["jobId"] == "job-456"
        assert otel["jobDetailsUri"] == "https://example.com/jobs/456"

    def test_api_workflow_tool_has_job_fields(self) -> None:
        """ApiWorkflowToolSpanAttributes should have jobId and jobDetailsUri."""
        attrs = ApiWorkflowToolSpanAttributes(
            tool_name="api_workflow",
            arguments={"endpoint": "/api/test"},
            job_id="job-789",
            job_details_uri="https://example.com/jobs/789",
        )
        otel = attrs.to_otel_attributes()

        assert otel["type"] == "apiWorkflowTool"
        assert otel["jobId"] == "job-789"

    def test_agentic_process_tool_has_job_fields(self) -> None:
        """AgenticProcessToolSpanAttributes should have jobId and jobDetailsUri."""
        attrs = AgenticProcessToolSpanAttributes(
            tool_name="agentic_process",
            arguments={"process": "test"},
            job_id="job-abc",
            job_details_uri="https://example.com/jobs/abc",
        )
        otel = attrs.to_otel_attributes()

        assert otel["type"] == "agenticProcessTool"
        assert otel["jobId"] == "job-abc"

    def test_escalation_tool_has_channel_fields(self) -> None:
        """EscalationToolSpanAttributes should have channel-specific fields."""
        attrs = EscalationToolSpanAttributes(
            arguments={"reason": "needs approval"},
            channel_type="teams",
            assigned_to="user@example.com",
            task_id="task-123",
            task_url="https://example.com/tasks/123",
        )
        otel = attrs.to_otel_attributes()

        assert otel["type"] == "escalationTool"
        assert otel["channelType"] == "teams"
        assert otel["assignedTo"] == "user@example.com"
        assert otel["taskId"] == "task-123"
        assert otel["taskUrl"] == "https://example.com/tasks/123"

    def test_integration_tool_has_tool_name(self) -> None:
        """IntegrationToolSpanAttributes should have toolName (unlike ProcessToolSpanAttributes)."""
        attrs = IntegrationToolSpanAttributes(
            tool_name="gmail_send",
            arguments={"to": "test@example.com"},
        )
        otel = attrs.to_otel_attributes()

        assert otel["type"] == "integrationTool"
        assert otel["toolName"] == "gmail_send"

    def test_action_center_tool_has_tool_name(self) -> None:
        """ActionCenterToolSpanAttributes should have toolName."""
        attrs = ActionCenterToolSpanAttributes(
            tool_name="create_task",
            arguments={"title": "Review document"},
        )
        otel = attrs.to_otel_attributes()

        assert otel["type"] == "actionCenterTool"
        assert otel["toolName"] == "create_task"


class TestCompletionToolCallsFormat:
    """Tests that CompletionSpanAttributes tool_calls format matches C# SDK."""

    def test_tool_calls_camel_case(self) -> None:
        """Tool calls in CompletionSpanAttributes should use camelCase."""
        attrs = CompletionSpanAttributes(
            model="gpt-4o",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="test_tool",
                    arguments={"param": "value"},
                )
            ],
        )
        otel = attrs.to_otel_attributes()
        tool_calls = otel.get("toolCalls")

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["name"] == "test_tool"
        assert tool_calls[0]["arguments"] == {"param": "value"}

    def test_multiple_tool_calls(self) -> None:
        """Multiple tool calls should all be serialized correctly."""
        attrs = CompletionSpanAttributes(
            model="gpt-4o",
            tool_calls=[
                ToolCall(id="call_1", name="tool_a", arguments={"a": 1}),
                ToolCall(id="call_2", name="tool_b", arguments={"b": 2}),
            ],
        )
        otel = attrs.to_otel_attributes()
        tool_calls = otel.get("toolCalls")

        assert tool_calls is not None
        assert len(tool_calls) == 2


class TestErrorDetailsFormat:
    """Tests that error details format matches C# SDK."""

    def test_error_details_camel_case(self) -> None:
        """ErrorDetails fields should use camelCase."""
        from uipath_agents._observability.llmops.spans.span_attributes.base import (
            ErrorDetails,
        )

        attrs = AgentRunSpanAttributes(
            agent_name="Test",
            agent_id="123",
            error=ErrorDetails(
                message="Test error",
                type="ValueError",
                stack_trace="Traceback...",
            ),
        )
        otel = attrs.to_otel_attributes()
        error = otel.get("error")

        assert error is not None
        assert error["message"] == "Test error"
        assert error["type"] == "ValueError"
        assert error["stackTrace"] == "Traceback..."
        assert "stack_trace" not in error


class TestNoneValuesExcluded:
    """Tests that None values are properly excluded from OTEL output."""

    def test_none_fields_excluded_from_agent_run(self) -> None:
        """None fields should not appear in OTEL output."""
        attrs = AgentRunSpanAttributes(
            agent_name="Test",
            agent_id="123",
            system_prompt=None,
            user_prompt=None,
            input=None,
            output=None,
        )
        otel = attrs.to_otel_attributes()

        assert "systemPrompt" not in otel
        assert "userPrompt" not in otel
        assert "input" not in otel
        assert "output" not in otel

    def test_none_fields_excluded_from_tool_call(self) -> None:
        """None fields should not appear in ToolCallSpanAttributes output."""
        attrs = ToolCallSpanAttributes(
            tool_name="test",
            call_id=None,
            arguments=None,
            result=None,
        )
        otel = attrs.to_otel_attributes()

        assert "callId" not in otel
        assert "arguments" not in otel
        assert "result" not in otel
