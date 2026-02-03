"""Unit tests for span attribute serialization and JSON output."""

from datetime import datetime, timezone
from typing import Any, Dict

from uipath_agents._observability.llmops.spans.span_attributes import (
    AgenticProcessToolSpanAttributes,
    AgentOutputSpanAttributes,
    AgentRunSpanAttributes,
    AgentToolSpanAttributes,
    ApiWorkflowToolSpanAttributes,
    CompletionSpanAttributes,
    EscalationToolSpanAttributes,
    IntegrationToolSpanAttributes,
    LlmCallSpanAttributes,
    ModelSpanAttributes,
    ProcessToolSpanAttributes,
    SpanType,
)


class TestBaseSpanAttributesLicenseRefId:
    """Tests for license_ref_id field on BaseSpanAttributes."""

    def test_license_ref_id_serializes_with_alias(self) -> None:
        attrs = ProcessToolSpanAttributes(
            tool_name="test_tool",
            license_ref_id="abc-123-def",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["licenseRefId"] == "abc-123-def"

    def test_license_ref_id_none_excluded(self) -> None:
        attrs = ProcessToolSpanAttributes(tool_name="test_tool")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert "licenseRefId" not in data


class TestEscalationToolSpanAttributesMemoryFields:
    """Tests for from_memory and saved_to_memory fields."""

    def test_from_memory_serializes_with_alias(self) -> None:
        attrs = EscalationToolSpanAttributes(from_memory=True)
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["fromMemory"] is True

    def test_saved_to_memory_serializes_with_alias(self) -> None:
        attrs = EscalationToolSpanAttributes(saved_to_memory=False)
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["savedToMemory"] is False

    def test_memory_fields_optional(self) -> None:
        attrs = EscalationToolSpanAttributes()
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert "fromMemory" not in data
        assert "savedToMemory" not in data


class TestIntegrationToolSpanAttributesArguments:
    """Tests for arguments field on IntegrationToolSpanAttributes."""

    def test_arguments_serializes_with_alias(self) -> None:
        args: Dict[str, Any] = {"key": "value", "count": 42}
        attrs = IntegrationToolSpanAttributes(tool_name="test", arguments=args)
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["arguments"] == args

    def test_arguments_optional(self) -> None:
        attrs = IntegrationToolSpanAttributes(tool_name="test")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert "arguments" not in data


class TestModelSpanAttributes:
    """Tests for ModelSpanAttributes class."""

    def test_is_deprecated_default_false(self) -> None:
        attrs = ModelSpanAttributes()
        assert attrs.is_deprecated is False

    def test_is_deprecated_serializes_with_alias(self) -> None:
        attrs = ModelSpanAttributes(is_deprecated=True)
        data = attrs.model_dump(by_alias=True)
        assert data["isDeprecated"] is True

    def test_retire_date_serializes_with_alias(self) -> None:
        dt = datetime(2025, 6, 15, tzinfo=timezone.utc)
        attrs = ModelSpanAttributes(retire_date=dt)
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["retireDate"] == dt

    def test_retire_date_optional(self) -> None:
        attrs = ModelSpanAttributes()
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert "retireDate" not in data


class TestCompletionSpanAttributesNewFields:
    """Tests for content, explanation, attributes fields on CompletionSpanAttributes."""

    def test_content_serializes_with_alias(self) -> None:
        attrs = CompletionSpanAttributes(content="Hello world")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["content"] == "Hello world"

    def test_explanation_serializes_with_alias(self) -> None:
        attrs = CompletionSpanAttributes(explanation="Reasoning here")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["explanation"] == "Reasoning here"

    def test_attributes_serializes_with_alias(self) -> None:
        model_attrs = ModelSpanAttributes(is_deprecated=True)
        attrs = CompletionSpanAttributes(attributes=model_attrs)
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["attributes"]["isDeprecated"] is True

    def test_new_fields_optional(self) -> None:
        attrs = CompletionSpanAttributes()
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert "content" not in data
        assert "explanation" not in data
        assert "attributes" not in data


class TestLlmCallSpanAttributesNewFields:
    """Tests for input, content, explanation fields on LlmCallSpanAttributes."""

    def test_input_serializes_with_alias(self) -> None:
        attrs = LlmCallSpanAttributes(input="User message")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["input"] == "User message"

    def test_content_serializes_with_alias(self) -> None:
        attrs = LlmCallSpanAttributes(content="Response content")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["content"] == "Response content"

    def test_explanation_serializes_with_alias(self) -> None:
        attrs = LlmCallSpanAttributes(explanation="Why this response")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["explanation"] == "Why this response"

    def test_new_fields_optional(self) -> None:
        attrs = LlmCallSpanAttributes()
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert "input" not in data
        assert "content" not in data
        assert "explanation" not in data


class TestProcessToolSpanAttributesJobFields:
    """Tests for job_id, job_details_uri on ProcessToolSpanAttributes."""

    def test_job_id_serializes_with_alias(self) -> None:
        attrs = ProcessToolSpanAttributes(tool_name="proc", job_id="job-123")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["jobId"] == "job-123"

    def test_job_details_uri_serializes_with_alias(self) -> None:
        attrs = ProcessToolSpanAttributes(
            tool_name="proc", job_details_uri="https://orch/jobs/123"
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["jobDetailsUri"] == "https://orch/jobs/123"

    def test_job_fields_optional(self) -> None:
        attrs = ProcessToolSpanAttributes(tool_name="proc")
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert "jobId" not in data
        assert "jobDetailsUri" not in data


class TestAgentToolSpanAttributes:
    """Tests for AgentToolSpanAttributes class."""

    def test_type_is_agent_tool(self) -> None:
        attrs = AgentToolSpanAttributes(tool_name="agent")
        assert attrs.type == SpanType.AGENT_TOOL

    def test_job_fields_serialize(self) -> None:
        attrs = AgentToolSpanAttributes(
            tool_name="agent",
            job_id="agent-job-1",
            job_details_uri="https://orch/agent/1",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["jobId"] == "agent-job-1"
        assert data["jobDetailsUri"] == "https://orch/agent/1"


class TestApiWorkflowToolSpanAttributes:
    """Tests for ApiWorkflowToolSpanAttributes class."""

    def test_type_is_api_workflow_tool(self) -> None:
        attrs = ApiWorkflowToolSpanAttributes(tool_name="workflow")
        assert attrs.type == SpanType.API_WORKFLOW_TOOL

    def test_job_fields_serialize(self) -> None:
        attrs = ApiWorkflowToolSpanAttributes(
            tool_name="workflow",
            job_id="wf-job-1",
            job_details_uri="https://orch/workflow/1",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["jobId"] == "wf-job-1"
        assert data["jobDetailsUri"] == "https://orch/workflow/1"


class TestAgenticProcessToolSpanAttributes:
    """Tests for AgenticProcessToolSpanAttributes class."""

    def test_type_is_agentic_process_tool(self) -> None:
        attrs = AgenticProcessToolSpanAttributes(tool_name="agentic")
        assert attrs.type == SpanType.AGENTIC_PROCESS_TOOL

    def test_job_fields_serialize(self) -> None:
        attrs = AgenticProcessToolSpanAttributes(
            tool_name="agentic",
            job_id="ap-job-1",
            job_details_uri="https://orch/agentic/1",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)
        assert data["jobId"] == "ap-job-1"
        assert data["jobDetailsUri"] == "https://orch/agentic/1"


# ---------------------------------------------------------------------------
# JSON Serialization Tests
# ---------------------------------------------------------------------------


class TestCSharpSchemaCompatibility:
    """Tests verifying JSON output field names and serialization."""

    def test_process_tool_json_matches_csharp_schema(self) -> None:
        """ProcessToolSpanAttributes JSON serialization uses camelCase field names."""
        attrs = ProcessToolSpanAttributes(
            tool_name="InvoiceProcessor",
            job_id="12345678-abcd-1234-abcd-123456789012",
            job_details_uri="https://cloud.uipath.com/org/tenant/jobs/12345",
            license_ref_id="lic-ref-001",
            arguments={"invoiceId": "INV-001"},
            result={"status": "processed"},
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        # ProcessToolSpanAttributes expected fields
        assert "toolName" in data
        assert "jobId" in data
        assert "jobDetailsUri" in data
        assert "licenseRefId" in data
        assert "arguments" in data
        assert "result" in data

        # Verify no snake_case leaked through
        assert "tool_name" not in data
        assert "job_id" not in data
        assert "job_details_uri" not in data
        assert "license_ref_id" not in data

    def test_escalation_tool_json_matches_csharp_schema(self) -> None:
        """EscalationToolSpanAttributes JSON serialization uses camelCase field names."""
        attrs = EscalationToolSpanAttributes(
            channel_type="ActionCenter",
            assigned_to="user@example.com",
            task_id="task-123",
            task_url="https://cloud.uipath.com/tasks/123",
            from_memory=True,
            saved_to_memory=False,
            license_ref_id="lic-ref-002",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        # EscalationToolSpanAttributes expected fields
        assert "channelType" in data
        assert "assignedTo" in data
        assert "taskId" in data
        assert "taskUrl" in data
        assert "fromMemory" in data
        assert "savedToMemory" in data
        assert "licenseRefId" in data

    def test_completion_span_json_matches_csharp_schema(self) -> None:
        """CompletionSpanAttributes JSON serialization uses camelCase field names."""
        model_attrs = ModelSpanAttributes(
            is_deprecated=True,
            retire_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        attrs = CompletionSpanAttributes(
            model="gpt-4",
            content="Response text",
            explanation="Reasoning explanation",
            attributes=model_attrs,
            license_ref_id="lic-ref-003",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        # CompletionSpanAttributes expected fields
        assert "model" in data
        assert "content" in data
        assert "explanation" in data
        assert "attributes" in data
        assert "licenseRefId" in data

        # Nested ModelSpanAttributes
        assert data["attributes"]["isDeprecated"] is True
        assert "retireDate" in data["attributes"]

    def test_llm_call_span_json_matches_csharp_schema(self) -> None:
        """LlmCallSpanAttributes JSON serialization uses camelCase field names."""
        attrs = LlmCallSpanAttributes(
            input="User query",
            content="LLM response",
            explanation="Why this was generated",
            license_ref_id="lic-ref-004",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        # LlmCallSpanAttributes expected fields
        assert "input" in data
        assert "content" in data
        assert "explanation" in data
        assert "licenseRefId" in data

    def test_integration_tool_json_matches_csharp_schema(self) -> None:
        """IntegrationToolSpanAttributes JSON serialization uses camelCase field names."""
        attrs = IntegrationToolSpanAttributes(
            tool_name="Salesforce_CreateAccount",
            arguments={"accountName": "Acme Corp"},
            result={"accountId": "001xx000003DGbY"},
            license_ref_id="lic-ref-005",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        # IntegrationToolSpanAttributes expected fields
        assert "toolName" in data
        assert "arguments" in data
        assert "result" in data
        assert "licenseRefId" in data

    def test_agent_tool_json_matches_csharp_schema(self) -> None:
        """AgentToolSpanAttributes JSON serialization uses camelCase field names."""
        attrs = AgentToolSpanAttributes(
            tool_name="SubAgent",
            job_id="agent-job-12345",
            job_details_uri="https://cloud.uipath.com/jobs/agent/12345",
            license_ref_id="lic-ref-006",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        assert "toolName" in data
        assert "jobId" in data
        assert "jobDetailsUri" in data
        assert "licenseRefId" in data
        assert attrs.type == SpanType.AGENT_TOOL

    def test_api_workflow_tool_json_matches_csharp_schema(self) -> None:
        """ApiWorkflowToolSpanAttributes JSON serialization uses camelCase field names."""
        attrs = ApiWorkflowToolSpanAttributes(
            tool_name="WorkflowExecutor",
            job_id="wf-job-12345",
            job_details_uri="https://cloud.uipath.com/jobs/workflow/12345",
            license_ref_id="lic-ref-007",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        assert "toolName" in data
        assert "jobId" in data
        assert "jobDetailsUri" in data
        assert "licenseRefId" in data
        assert attrs.type == SpanType.API_WORKFLOW_TOOL

    def test_agentic_process_tool_json_matches_csharp_schema(self) -> None:
        """AgenticProcessToolSpanAttributes JSON serialization uses camelCase field names."""
        attrs = AgenticProcessToolSpanAttributes(
            tool_name="AgenticProcess",
            job_id="ap-job-12345",
            job_details_uri="https://cloud.uipath.com/jobs/agentic/12345",
            license_ref_id="lic-ref-008",
        )
        data = attrs.model_dump(by_alias=True, exclude_none=True)

        assert "toolName" in data
        assert "jobId" in data
        assert "jobDetailsUri" in data
        assert "licenseRefId" in data
        assert attrs.type == SpanType.AGENTIC_PROCESS_TOOL


# ---------------------------------------------------------------------------
# AgentRun Parity Tests - Critical for LLMOps dashboard
# ---------------------------------------------------------------------------


class TestAgentRunSpanAttributes:
    """Tests for AgentRunSpanAttributes serialization and field handling."""

    def test_type_is_agent_run(self) -> None:
        """Verify type property returns correct span type."""
        attrs = AgentRunSpanAttributes(agent_name="TestAgent")
        assert attrs.type == SpanType.AGENT_RUN

    def test_is_conversational_false_not_excluded(self) -> None:
        """CRITICAL: isConversational=False must appear in output, not be excluded.

        Regression test for bug where False was converted to None and excluded.
        """
        attrs = AgentRunSpanAttributes(
            agent_name="TestAgent",
            is_conversational=False,
        )
        data = attrs.to_otel_attributes()

        assert "isConversational" in data, (
            "isConversational=False should not be excluded"
        )
        assert data["isConversational"] is False

    def test_is_conversational_true_included(self) -> None:
        """isConversational=True should be included."""
        attrs = AgentRunSpanAttributes(
            agent_name="TestAgent",
            is_conversational=True,
        )
        data = attrs.to_otel_attributes()

        assert data["isConversational"] is True

    def test_is_conversational_none_excluded(self) -> None:
        """isConversational=None should be excluded (optional field)."""
        attrs = AgentRunSpanAttributes(agent_name="TestAgent")
        data = attrs.to_otel_attributes()

        assert "isConversational" not in data

    def test_output_serializes_correctly(self) -> None:
        """output field should serialize with alias."""
        attrs = AgentRunSpanAttributes(
            agent_name="TestAgent",
            output={"result": "success", "value": 42},
        )
        data = attrs.to_otel_attributes()

        assert "output" in data
        assert data["output"] == {"result": "success", "value": 42}

    def test_required_attributes_present(self) -> None:
        """Verify all required attributes are present in to_otel_attributes()."""
        attrs = AgentRunSpanAttributes(
            agent_name="TestAgent",
            agent_id="agent-123",
            is_conversational=False,
            system_prompt="You are helpful",
            user_prompt="Hello",
            input={"query": "test"},
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            output={"content": "response"},
            execution_type=1,
        )
        data = attrs.to_otel_attributes()

        # Required attributes
        assert data["type"] == "agentRun"
        assert data["agentName"] == "TestAgent"
        assert data["agentId"] == "agent-123"
        assert data["isConversational"] is False
        assert data["systemPrompt"] == "You are helpful"
        assert data["userPrompt"] == "Hello"
        assert data["input"] == {"query": "test"}
        assert data["inputSchema"] == {"type": "object"}
        assert data["outputSchema"] == {"type": "object"}
        assert data["output"] == {"content": "response"}
        assert data["source"] == "unknown"
        assert data["uipath.source"] == 1  # SourceEnum.Agents

    def test_no_snake_case_in_output(self) -> None:
        """Verify no snake_case field names leak into JSON output."""
        attrs = AgentRunSpanAttributes(
            agent_name="TestAgent",
            agent_id="123",
            is_conversational=True,
            system_prompt="sys",
            user_prompt="usr",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            execution_type=1,
        )
        data = attrs.to_otel_attributes()

        # These snake_case names should NOT appear
        assert "agent_name" not in data
        assert "agent_id" not in data
        assert "is_conversational" not in data
        assert "system_prompt" not in data
        assert "user_prompt" not in data
        assert "input_schema" not in data
        assert "output_schema" not in data
        assert "execution_type" not in data
        assert "uipath_source" not in data


class TestAgentOutputSpanAttributes:
    """Tests for AgentOutputSpanAttributes parity."""

    def test_type_is_agent_output(self) -> None:
        """Verify type property returns correct span type."""
        attrs = AgentOutputSpanAttributes()
        assert attrs.type == SpanType.AGENT_OUTPUT

    def test_output_serializes_correctly(self) -> None:
        """output field should serialize."""
        attrs = AgentOutputSpanAttributes(output="The answer is 42")
        data = attrs.to_otel_attributes()

        assert "output" in data
        assert data["output"] == "The answer is 42"
