"""Guardrail-related span attribute classes."""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from .base import BaseSpanAttributes
from .types import SpanType


class GuardrailEvaluationSpanAttributes(BaseSpanAttributes):
    """Attributes for guardrail evaluation spans."""

    model_config = ConfigDict(populate_by_name=True)

    guardrail_name: str = Field(..., alias="guardrailName")
    guardrail_description: Optional[str] = Field(None, alias="guardrailDescription")
    validation_result: Optional[str] = Field(None, alias="validationResult")
    guardrail_action: Optional[str] = Field(None, alias="guardrailAction")

    # Additional evaluation fields
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
    """Attributes for LLM pre-guardrails spans."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.LLM_PRE_GUARDRAILS


class LlmPostGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for LLM post-guardrails spans."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.LLM_POST_GUARDRAILS


class ToolPreGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for tool pre-guardrails spans."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.TOOL_PRE_GUARDRAILS


class ToolPostGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for tool post-guardrails spans."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.TOOL_POST_GUARDRAILS


class AgentPreGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for agent pre-guardrails spans."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.AGENT_PRE_GUARDRAILS


class AgentPostGuardrailsSpanAttributes(BaseSpanAttributes):
    """Attributes for agent post-guardrails spans."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def type(self) -> str:
        return SpanType.AGENT_POST_GUARDRAILS


class GuardrailEscalationSpanAttributes(BaseSpanAttributes):
    """Attributes for guardrail escalation spans."""

    model_config = ConfigDict(populate_by_name=True)

    guardrail_name: str = Field(..., alias="guardrailName")
    guardrail_description: Optional[str] = Field(None, alias="guardrailDescription")
    guardrail_action: Optional[str] = Field(None, alias="guardrailAction")
    details: Optional[Any] = Field(None, alias="details")
    action: Optional[str] = Field(None, alias="action")
    reason: Optional[str] = Field(None, alias="reason")
    severity_level: Optional[str] = Field(None, alias="severityLevel")
    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    task_arguments: Optional[Any] = Field(None, alias="taskArguments")
    assigned_to: Optional[str] = Field(None, alias="assignedTo")
    updated_arguments: Optional[Any] = Field(None, alias="updatedArguments")
    task_url: Optional[str] = Field(None, alias="taskUrl")
    review_status: Optional[str] = Field(None, alias="reviewStatus")
    reviewed_by: Optional[str] = Field(None, alias="reviewedBy")
    review_outcome: Optional[str] = Field(None, alias="reviewOutcome")
    review_reason: Optional[Any] = Field(None, alias="reviewReason")
    reviewed_inputs: Optional[Any] = Field(None, alias="reviewedInputs")
    reviewed_outputs: Optional[Any] = Field(None, alias="reviewedOutputs")

    @property
    def type(self) -> str:
        return SpanType.GUARDRAIL_ESCALATION


class ToolGuardrailEvaluationSpanAttributes(BaseSpanAttributes):
    """Attributes for tool guardrail evaluation spans."""

    model_config = ConfigDict(populate_by_name=True)

    guardrail_name: str = Field(..., alias="guardrailName")
    guardrail_description: Optional[str] = Field(None, alias="guardrailDescription")
    guardrail_action: Optional[str] = Field(None, alias="guardrailAction")
    details: Optional[Any] = Field(None, alias="details")
    action: Optional[str] = Field(None, alias="action")
    arguments: Optional[Dict[str, Any]] = Field(None, alias="arguments")
    result: Optional[Any] = Field(None, alias="result")
    assigned_to: Optional[str] = Field(None, alias="assignedTo")
    updated_arguments: Optional[Any] = Field(None, alias="updatedArguments")
    updated_result: Optional[Any] = Field(None, alias="updatedResult")
    excluded_fields: Optional[List[str]] = Field(None, alias="excludedFields")
    severity_level: Optional[str] = Field(None, alias="severityLevel")
    reason: Optional[str] = Field(None, alias="reason")

    @property
    def type(self) -> str:
        return SpanType.TOOL_GUARDRAIL_EVALUATION


class ToolGuardrailEscalationSpanAttributes(ToolGuardrailEvaluationSpanAttributes):
    """Attributes for tool guardrail escalation spans."""

    task_arguments: Optional[Any] = Field(None, alias="taskArguments")
    task_url: Optional[str] = Field(None, alias="taskUrl")
    review_status: Optional[str] = Field(None, alias="reviewStatus")
    reviewed_by: Optional[str] = Field(None, alias="reviewedBy")
    review_outcome: Optional[str] = Field(None, alias="reviewOutcome")
    review_reason: Optional[Any] = Field(None, alias="reviewReason")
    reviewed_inputs: Optional[Any] = Field(None, alias="reviewedInputs")
    reviewed_outputs: Optional[Any] = Field(None, alias="reviewedOutputs")

    @property
    def type(self) -> str:
        return SpanType.TOOL_GUARDRAIL_ESCALATION
