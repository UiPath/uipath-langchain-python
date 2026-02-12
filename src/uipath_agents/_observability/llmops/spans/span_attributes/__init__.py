"""Typed span attribute classes matching UiPath LLMOPs schema.

This package provides type-safe span attribute handling matching
"""

from typing import Union

from .agent import (
    AgentInputSpanAttributes,
    AgentOutputSpanAttributes,
    AgentRunSpanAttributes,
)
from .base import BaseSpanAttributes, ErrorDetails
from .context import ContextGroundingToolSpanAttributes
from .governance import (
    GovernanceSpanAttributes,
    PostGovernanceSpanAttributes,
    PreGovernanceSpanAttributes,
    ToolPostGovernanceSpanAttributes,
    ToolPreGovernanceSpanAttributes,
)
from .guardrails import (
    AgentPostGuardrailsSpanAttributes,
    AgentPreGuardrailsSpanAttributes,
    GuardrailEscalationSpanAttributes,
    GuardrailEvaluationSpanAttributes,
    LlmPostGuardrailsSpanAttributes,
    LlmPreGuardrailsSpanAttributes,
    ToolGuardrailEscalationSpanAttributes,
    ToolGuardrailEvaluationSpanAttributes,
    ToolPostGuardrailsSpanAttributes,
    ToolPreGuardrailsSpanAttributes,
)
from .llm import (
    CompletionSpanAttributes,
    LlmCallSpanAttributes,
    ModelSettings,
    ModelSpanAttributes,
    ToolCall,
    Usage,
)
from .mcp import (
    McpSessionStartSpanAttributes,
    McpSessionStopSpanAttributes,
    McpToolSpanAttributes,
)
from .tool_call import ToolCallSpanAttributes, ToolExecutionSpanAttributes
from .tools import (
    ActionCenterToolSpanAttributes,
    AgenticProcessToolSpanAttributes,
    AgentToolSpanAttributes,
    ApiWorkflowToolSpanAttributes,
    EscalationToolSpanAttributes,
    IntegrationToolSpanAttributes,
    InternalToolSpanAttributes,
    IxpToolSpanAttributes,
    ProcessToolSpanAttributes,
    VsEscalationToolSpanAttributes,
)
from .types import (
    ENV_UIPATH_IS_DEBUG,
    ENV_UIPATH_PROCESS_VERSION,
    ExecutionType,
    SpanType,
    get_agent_version,
    get_execution_type,
)

__all__ = [
    # Types and constants
    "SpanType",
    "ExecutionType",
    "ENV_UIPATH_IS_DEBUG",
    "ENV_UIPATH_PROCESS_VERSION",
    "get_execution_type",
    "get_agent_version",
    # Base classes
    "BaseSpanAttributes",
    "ErrorDetails",
    # Core
    "AgentRunSpanAttributes",
    "CompletionSpanAttributes",
    "LlmCallSpanAttributes",
    "ToolCallSpanAttributes",
    "ToolExecutionSpanAttributes",
    "AgentOutputSpanAttributes",
    "AgentInputSpanAttributes",
    "ModelSettings",
    "ModelSpanAttributes",
    "Usage",
    "ToolCall",
    # Tools
    "ProcessToolSpanAttributes",
    "ActionCenterToolSpanAttributes",
    "AgentToolSpanAttributes",
    "ApiWorkflowToolSpanAttributes",
    "AgenticProcessToolSpanAttributes",
    "EscalationToolSpanAttributes",
    "IntegrationToolSpanAttributes",
    "InternalToolSpanAttributes",
    "IxpToolSpanAttributes",
    "VsEscalationToolSpanAttributes",
    # Guardrails
    "GuardrailEvaluationSpanAttributes",
    "LlmPreGuardrailsSpanAttributes",
    "LlmPostGuardrailsSpanAttributes",
    "ToolPreGuardrailsSpanAttributes",
    "ToolPostGuardrailsSpanAttributes",
    "AgentPreGuardrailsSpanAttributes",
    "AgentPostGuardrailsSpanAttributes",
    "GuardrailEscalationSpanAttributes",
    "ToolGuardrailEvaluationSpanAttributes",
    "ToolGuardrailEscalationSpanAttributes",
    # MCP
    "McpToolSpanAttributes",
    "McpSessionStartSpanAttributes",
    "McpSessionStopSpanAttributes",
    # Context Grounding
    "ContextGroundingToolSpanAttributes",
    # Governance
    "GovernanceSpanAttributes",
    "PreGovernanceSpanAttributes",
    "PostGovernanceSpanAttributes",
    "ToolPreGovernanceSpanAttributes",
    "ToolPostGovernanceSpanAttributes",
    # Type alias
    "SpanAttributes",
]

# Type alias for all span attribute types
SpanAttributes = Union[
    AgentRunSpanAttributes,
    CompletionSpanAttributes,
    LlmCallSpanAttributes,
    ToolCallSpanAttributes,
    ProcessToolSpanAttributes,
    EscalationToolSpanAttributes,
    IntegrationToolSpanAttributes,
    InternalToolSpanAttributes,
    IxpToolSpanAttributes,
    VsEscalationToolSpanAttributes,
    AgentOutputSpanAttributes,
    GuardrailEvaluationSpanAttributes,
    LlmPreGuardrailsSpanAttributes,
    LlmPostGuardrailsSpanAttributes,
    ToolPreGuardrailsSpanAttributes,
    ToolPostGuardrailsSpanAttributes,
    AgentPreGuardrailsSpanAttributes,
    AgentPostGuardrailsSpanAttributes,
    McpToolSpanAttributes,
    McpSessionStartSpanAttributes,
    McpSessionStopSpanAttributes,
    ContextGroundingToolSpanAttributes,
    PreGovernanceSpanAttributes,
    PostGovernanceSpanAttributes,
    ToolPreGovernanceSpanAttributes,
    ToolPostGovernanceSpanAttributes,
    GuardrailEscalationSpanAttributes,
    ToolGuardrailEvaluationSpanAttributes,
    ToolGuardrailEscalationSpanAttributes,
]
