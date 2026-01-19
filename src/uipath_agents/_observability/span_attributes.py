"""Typed span attribute classes matching UiPath Temporal schema.

These classes provide type-safe span attribute handling matching
the Temporal implementation in agents/backend/Execution.Shared/Traces/.
"""

import os
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# --- Environment Variable Names ---
ENV_UIPATH_IS_DEBUG = "UIPATH_IS_DEBUG"
ENV_UIPATH_PROCESS_VERSION = "UIPATH_PROCESS_VERSION"


class ExecutionType(IntEnum):
    """Execution type matching Temporal/Orchestrator schema.

    Debug (0): Studio debug, playground, local development
    Runtime (1): Production job from Orchestrator
    """

    DEBUG = 0
    RUNTIME = 1


def get_execution_type() -> int:
    """Get execution type from environment.

    Returns:
        ExecutionType.DEBUG (0) if UIPATH_IS_DEBUG=true
        ExecutionType.RUNTIME (1) if UIPATH_IS_DEBUG=false or not set
    """
    env_value = os.getenv(ENV_UIPATH_IS_DEBUG, "").lower()
    if env_value == "true":
        return ExecutionType.DEBUG
    return ExecutionType.RUNTIME


def get_agent_version() -> Optional[str]:
    """Get agent version from environment."""
    return os.getenv(ENV_UIPATH_PROCESS_VERSION) or None


class SpanType:
    """Span type constants matching Temporal SpanType."""

    # Core types
    AGENT_RUN = "agentRun"
    COMPLETION = "completion"
    LLM_CALL = "llmCall"
    TOOL_CALL = "toolCall"
    TOOL_EXECUTION = "toolExecution"
    AGENT_OUTPUT = "agentOutput"
    AGENT_INPUT = "agentInput"
    ACTION_CENTER_TOOL = "actionCenterTool"

    # Tool types
    PROCESS_TOOL = "processTool"
    AGENT_TOOL = "agentTool"
    API_WORKFLOW_TOOL = "apiWorkflowTool"
    AGENTIC_PROCESS_TOOL = "agenticProcessTool"
    INTEGRATION_TOOL = "integrationTool"
    CONTEXT_GROUNDING_TOOL = "contextGroundingTool"
    ESCALATION_TOOL = "escalationTool"
    MCP_TOOL = "mcpTool"
    MOCK_TOOL = "mockTool"
    IXP_TOOL = "ixpTool"
    INTERNAL_TOOL = "internalTool"

    # Guardrail types (matching Temporal exactly)
    TOOL_PRE_GUARDRAILS = "toolPreGuardrails"
    TOOL_POST_GUARDRAILS = "toolPostGuardrails"
    LLM_PRE_GUARDRAILS = "llmPreGuardrails"
    LLM_POST_GUARDRAILS = "llmPostGuardrails"
    AGENT_PRE_GUARDRAILS = "agentPreGuardrails"
    AGENT_POST_GUARDRAILS = "agentPostGuardrails"
    GUARDRAIL_EVALUATION = "guardrailEvaluation"
    GUARDRAIL_ESCALATION = "guardrailEscalation"

    # Governance types
    TOOL_PRE_GOVERNANCE = "toolPreGovernance"
    TOOL_POST_GOVERNANCE = "toolPostGovernance"
    PRE_GOVERNANCE = "preGovernance"
    POST_GOVERNANCE = "postGovernance"
    GOVERNANCE_ESCALATION = "governanceEscalation"

    # MCP types
    MCP_SESSION_START = "mcpSessionStart"
    MCP_SESSION_STOP = "mcpSessionStop"

    # Eval types
    EVAL_SET_RUN = "evalSetRun"
    EVAL = "eval"
    EVAL_ASSERTIONS = "evalAssertions"
    EVAL_OUTPUT = "evalOutput"

    # Other types
    PLANNING = "planning"
    VALIDATION = "validation"
    OUTPUT_CORRECTION = "outputCorrection"
    SIMULATED_TOOL = "simulatedTool"
    SIMULATED_INPUT = "simulatedInput"
    AGENT_MEMORY_LOOKUP = "agentMemoryLookup"
    APPLY_DYNAMIC_FEW_SHOT = "applyDynamicFewShot"


class ErrorDetails(BaseModel):
    """Error details structure matching Temporal ErrorWithDetails."""

    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(..., alias="message")
    type: str = Field(..., alias="type")
    stack_trace: Optional[str] = Field(None, alias="stackTrace")


class BaseSpanAttributes(BaseModel, ABC):
    """Abstract base class for all span attributes.

    Matches Temporal BaseSpanAttributes with polymorphic JSON serialization.
    Each subclass must implement the `type` property.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",  # Allow extra fields for forward compatibility
    )

    error: Optional[ErrorDetails] = Field(None, alias="error")

    @property
    @abstractmethod
    def type(self) -> str: ...

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OpenTelemetry attribute dict.

        Returns dict with both 'type' and 'span_type' (for backend compatibility).
        None values excluded. Complex objects kept as-is for span processor.
        """
        # Include both 'type' (Temporal schema) and 'span_type' (backend extraction)
        attrs: Dict[str, Any] = {"type": self.type, "span_type": self.type}
        data = self.model_dump(by_alias=True, exclude_none=True, exclude={"error"})
        attrs.update(data)

        if self.error:
            attrs["error"] = self.error.model_dump(by_alias=True)
        return attrs


class AgentRunSpanAttributes(BaseSpanAttributes):
    """Attributes for agent run spans.

    Matches Temporal AgentRunSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    agent_id: Optional[str] = Field(None, alias="agentId")
    agent_name: str = Field(..., alias="agentName")
    source: str = Field(default="langchain", alias="source")
    is_conversational: Optional[bool] = Field(None, alias="isConversational")
    system_prompt: Optional[str] = Field(None, alias="systemPrompt")
    user_prompt: Optional[str] = Field(None, alias="userPrompt")
    input_schema: Optional[Dict[str, Any]] = Field(None, alias="inputSchema")
    output_schema: Optional[Dict[str, Any]] = Field(None, alias="outputSchema")
    input: Optional[Dict[str, Any]] = Field(None, alias="input")
    output: Optional[Any] = Field(None, alias="output")

    # Execution context fields
    execution_type: Optional[int] = Field(None, alias="executionType")
    agent_version: Optional[str] = Field(None, alias="agentVersion")

    @property
    def type(self) -> str:
        return SpanType.AGENT_RUN


class ModelSettings(BaseModel):
    """Model settings matching Temporal ModelSpanSettings."""

    model_config = ConfigDict(populate_by_name=True)

    max_tokens: Optional[int] = Field(None, alias="maxTokens")
    temperature: Optional[float] = Field(None, alias="temperature")


class Usage(BaseModel):
    """Token usage matching Temporal Usage record."""

    model_config = ConfigDict(populate_by_name=True)

    completion_tokens: int = Field(..., alias="completionTokens")
    prompt_tokens: int = Field(..., alias="promptTokens")
    total_tokens: int = Field(..., alias="totalTokens")
    is_byo_execution: bool = Field(False, alias="isByoExecution")
    execution_deployment_type: Optional[str] = Field(
        None, alias="executionDeploymentType"
    )
    is_pii_masked: bool = Field(False, alias="isPiiMasked")
    llm_calls: int = Field(1, alias="llmCalls")


class ToolCall(BaseModel):
    """Tool call matching Temporal ToolCall record."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="id")
    name: str = Field(..., alias="name")
    arguments: Dict[str, Any] = Field(..., alias="arguments")


class CompletionSpanAttributes(BaseSpanAttributes):
    """Attributes for Model run spans.

    Maps to Temporal CompletionSpanAttributes with type="completion".
    """

    model_config = ConfigDict(populate_by_name=True)

    model: Optional[str] = Field(None, alias="model")
    # Settings as nested object (matches Temporal schema)
    settings: Optional["ModelSettings"] = Field(None, alias="settings")
    tool_calls: Optional[List["ToolCall"]] = Field(None, alias="toolCalls")
    usage: Optional["Usage"] = Field(None, alias="usage")

    @property
    def type(self) -> str:
        return SpanType.COMPLETION


class LlmCallSpanAttributes(BaseSpanAttributes):
    """Attributes for LLM call spans (outer wrapper, no model)."""

    model_config = ConfigDict(populate_by_name=True)

    # Settings as nested object (matches Temporal)
    settings: Optional["ModelSettings"] = Field(None, alias="settings")

    @property
    def type(self) -> str:
        return SpanType.LLM_CALL


class ToolCallSpanAttributes(BaseSpanAttributes):
    """Attributes for tool call spans."""

    model_config = ConfigDict(populate_by_name=True)

    call_id: Optional[str] = Field(None, alias="callId")
    tool_name: str = Field(..., alias="toolName")
    tool_type: str = Field(default="toolCall", alias="toolType")
    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    result: Optional[Any] = Field(None, alias="result")
    # Internal field for span type override (not serialized)
    _span_type: Optional[str] = None

    def __init__(self, span_type: Optional[str] = None, **data: Any):
        super().__init__(**data)
        self._span_type = span_type

    @property
    def type(self) -> str:
        return self._span_type or self.tool_type


class ToolExecutionSpanAttributes(BaseSpanAttributes):
    """Attributes for tool execution spans (inner execution)."""

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(..., alias="toolName")

    @property
    def type(self) -> str:
        return SpanType.TOOL_EXECUTION


class ProcessToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for UiPath process tool calls."""

    @property
    def type(self) -> str:
        return SpanType.PROCESS_TOOL


class ActionCenterToolSpanAttributes(ToolCallSpanAttributes):
    """Attributes for Action Center tool calls."""

    @property
    def type(self) -> str:
        return SpanType.ACTION_CENTER_TOOL


class AgentOutputSpanAttributes(BaseSpanAttributes):
    """Attributes for agent output spans."""

    model_config = ConfigDict(populate_by_name=True)

    output: Optional[str] = Field(None, alias="output")

    @property
    def type(self) -> str:
        return SpanType.AGENT_OUTPUT


class AgentInputSpanAttributes(BaseSpanAttributes):
    """Attributes for agent input spans."""

    model_config = ConfigDict(populate_by_name=True)

    input: Optional[str] = Field(None, alias="input")

    @property
    def type(self) -> str:
        return SpanType.AGENT_INPUT


# ---------------------------------------------------------------------------
# Guardrail Span Attributes
# ---------------------------------------------------------------------------


class GuardrailEvaluationSpanAttributes(BaseSpanAttributes):
    """Attributes for guardrail evaluation spans.

    Matches Temporal GuardrailEvaluationSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    guardrail_name: str = Field(..., alias="guardrailName")
    guardrail_description: Optional[str] = Field(None, alias="guardrailDescription")
    validation_result: Optional[str] = Field(None, alias="validationResult")
    guardrail_action: Optional[str] = Field(None, alias="guardrailAction")

    # Additional fields matching Temporal
    details: Optional[Dict[str, Any]] = Field(None, alias="details")
    action: Optional[str] = Field(None, alias="action")
    payload: Optional[Dict[str, Any]] = Field(None, alias="payload")

    # Action = "Escalate"
    assigned_to: Optional[str] = Field(None, alias="assignedTo")

    # Action = "Filter"
    updated_input: Optional[Dict[str, Any]] = Field(None, alias="updatedInput")
    updated_output: Optional[Dict[str, Any]] = Field(None, alias="updatedOutput")
    excluded_fields: Optional[List[str]] = Field(None, alias="excludedFields")

    # Action = "Log"
    severity_level: Optional[str] = Field(None, alias="severityLevel")

    # Action = "Block" or "Skip"
    reason: Optional[str] = Field(None, alias="reason")

    @property
    def type(self) -> str:
        return SpanType.GUARDRAIL_EVALUATION


class LlmPreGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for LLM pre-guardrails spans.

    Matches Temporal LlmPreGuardrailsSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.LLM_PRE_GUARDRAILS


class LlmPostGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for LLM post-guardrails spans.

    Matches Temporal LlmPostGuardrailsSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.LLM_POST_GUARDRAILS


class ToolPreGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for tool pre-guardrails spans.

    Matches Temporal ToolPreGuardrailsSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.TOOL_PRE_GUARDRAILS


class ToolPostGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for tool post-guardrails spans.

    Matches Temporal ToolPostGuardrailsSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.TOOL_POST_GUARDRAILS


class AgentPreGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for agent pre-guardrails spans.

    Matches Temporal AgentPreGuardrailsSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.AGENT_PRE_GUARDRAILS


class AgentPostGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for agent post-guardrails spans.

    Matches Temporal AgentPostGuardrailsSpanAttributes.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.AGENT_POST_GUARDRAILS


class EscalationToolSpanAttributes(BaseSpanAttributes):
    """Attributes for escalation tool spans.

    Matches Temporal EscalationToolSpanAttributes.
    Child span under Tool call with app name.
    """

    model_config = ConfigDict(populate_by_name=True)

    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    channel_type: Optional[str] = Field(None, alias="channelType")
    assigned_to: Optional[str] = Field(None, alias="assignedTo")
    task_id: Optional[str] = Field(None, alias="taskId")
    task_url: Optional[str] = Field(None, alias="taskUrl")
    result: Optional[Any] = Field(None, alias="result")

    @property
    def type(self) -> str:
        return SpanType.ESCALATION_TOOL


class IntegrationToolSpanAttributes(BaseSpanAttributes):
    """Attributes for integration tool spans.

    Child span under Tool call for integration tool execution.
    Replaces the SDK's activity_invoke span which gets filtered.
    """

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(..., alias="toolName")
    result: Optional[Any] = Field(None, alias="result")

    @property
    def type(self) -> str:
        return SpanType.INTEGRATION_TOOL


# Type alias for all span attribute types
SpanAttributes = Union[
    AgentRunSpanAttributes,
    CompletionSpanAttributes,
    LlmCallSpanAttributes,
    ToolCallSpanAttributes,
    ProcessToolSpanAttributes,
    EscalationToolSpanAttributes,
    IntegrationToolSpanAttributes,
    AgentOutputSpanAttributes,
    GuardrailEvaluationSpanAttributes,
    LlmPreGuardrailsSpanAttributes,
    LlmPostGuardrailsSpanAttributes,
    ToolPreGuardrailsSpanAttributes,
    ToolPostGuardrailsSpanAttributes,
    AgentPreGuardrailsSpanAttributes,
    AgentPostGuardrailsSpanAttributes,
]
