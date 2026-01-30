"""LLM-related span attribute classes."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base import BaseSpanAttributes
from .types import SpanType


class ModelSettings(BaseModel):
    """Settings for language model requests."""

    model_config = ConfigDict(populate_by_name=True)

    max_tokens: Optional[int] = Field(None, alias="maxTokens")
    temperature: Optional[float] = Field(None, alias="temperature")


class Usage(BaseModel):
    """Token usage metrics for language model calls."""

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
    """Represents a single tool call made during execution."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="id")
    name: str = Field(..., alias="name")
    arguments: Dict[str, Any] = Field(..., alias="arguments")


class ModelSpanAttributes(BaseModel):
    """Model metadata attributes for deprecation tracking."""

    model_config = ConfigDict(populate_by_name=True)

    is_deprecated: bool = Field(False, alias="isDeprecated")
    retire_date: Optional[datetime] = Field(None, alias="retireDate")


class CompletionSpanAttributes(BaseSpanAttributes):
    """Attributes for model completion/LLM call spans."""

    model_config = ConfigDict(populate_by_name=True)

    model: Optional[str] = Field(None, alias="model")
    settings: Optional[ModelSettings] = Field(None, alias="settings")
    tool_calls: Optional[List[ToolCall]] = Field(None, alias="toolCalls")
    usage: Optional[Usage] = Field(None, alias="usage")
    content: Optional[str] = Field(None, alias="content")
    explanation: Optional[str] = Field(None, alias="explanation")
    attributes: Optional[ModelSpanAttributes] = Field(None, alias="attributes")

    @property
    def type(self) -> str:
        return SpanType.COMPLETION


class LlmCallSpanAttributes(BaseSpanAttributes):
    """Attributes for LLM call spans."""

    model_config = ConfigDict(populate_by_name=True)

    model: Optional[str] = Field(None, alias="model")
    settings: Optional[ModelSettings] = Field(None, alias="settings")
    input: Optional[str] = Field(None, alias="input")
    content: Optional[str] = Field(None, alias="content")
    explanation: Optional[str] = Field(None, alias="explanation")

    @property
    def type(self) -> str:
        return SpanType.COMPLETION
