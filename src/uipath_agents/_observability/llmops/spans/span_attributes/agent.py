"""Agent-related span attribute classes."""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field
from uipath.tracing import SpanAttachment

from .base import BaseSpanAttributes
from .types import SpanType


class AgentRunSpanAttributes(BaseSpanAttributes):
    """Attributes for agent run spans."""

    model_config = ConfigDict(populate_by_name=True)

    agent_id: Optional[str] = Field(None, alias="agentId")
    agent_name: str = Field(..., alias="agentName")
    source: str = Field(default="unknown", alias="source")
    is_conversational: Optional[bool] = Field(None, alias="isConversational")
    system_prompt: Optional[str] = Field(None, alias="systemPrompt")
    user_prompt: Optional[str] = Field(None, alias="userPrompt")
    input_schema: Optional[Dict[str, Any]] = Field(None, alias="inputSchema")
    output_schema: Optional[Dict[str, Any]] = Field(None, alias="outputSchema")
    input: Optional[Dict[str, Any]] = Field(None, alias="input")
    output: Optional[Any] = Field(None, alias="output")
    attachments: Optional[List[SpanAttachment]] = Field(None, alias="attachments")

    # Execution context fields (extracted to top-level by uipath.tracing)
    execution_type: Optional[int] = Field(None, alias="executionType")
    agent_version: Optional[str] = Field(None, alias="agentVersion")
    reference_id: Optional[str] = Field(None, alias="referenceId")
    uipath_source: int = Field(default=1, alias="uipath.source")  # SourceEnum.Agents

    @property
    def type(self) -> str:
        return SpanType.AGENT_RUN


class AgentOutputSpanAttributes(BaseSpanAttributes):
    """Attributes for agent output spans."""

    model_config = ConfigDict(populate_by_name=True)

    output: Optional[str] = Field(None, alias="output")
    attachments: Optional[List[SpanAttachment]] = Field(None, alias="attachments")

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
