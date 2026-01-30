"""LLMOps telemetry module for UiPath Agents.

This module contains all components for LLMOps traces (business value telemetry):
- Span factory for creating typed spans
- LangChain callback for instrumentation
- Span instrumentors for LLM, Tool, and Guardrail events
- Span attributes and schema definitions
"""

from .callback import LlmOpsInstrumentationCallback, get_current_run_id
from .instrumentors import (
    BaseSpanInstrumentor,
    GuardrailSpanInstrumentor,
    InstrumentationState,
    LlmSpanInstrumentor,
    ToolSpanInstrumentor,
)
from .span_filters import (
    is_azure_monitor_span,
    is_custom_instrumentation_span,
    is_http_instrumentation_span,
    is_openinference_span,
)
from .span_hierarchy import SpanHierarchyManager
from .spans import (
    AgentRunSpanAttributes,
    AgentToolSpanAttributes,
    CompletionSpanAttributes,
    ContextGroundingToolSpanAttributes,
    ErrorDetails,
    EscalationToolSpanAttributes,
    ExecutionType,
    GovernanceSpanAttributes,
    GuardrailEscalationSpanAttributes,
    GuardrailEvaluationSpanAttributes,
    IntegrationToolSpanAttributes,
    LlmCallSpanAttributes,
    LlmOpsSpanFactory,
    McpSessionStartSpanAttributes,
    McpToolSpanAttributes,
    ProcessToolSpanAttributes,
    SpanKeys,
    SpanType,
    ToolCallSpanAttributes,
    reference_id_context,
)
from .sqlite_trace_context_storage import SqliteTraceContextStorage
from .trace_context_storage import (
    PendingSpanData,
    TraceContextData,
    TraceContextStorage,
)

__all__ = [
    # Callback
    "LlmOpsInstrumentationCallback",
    "get_current_run_id",
    # Span factory
    "LlmOpsSpanFactory",
    # Instrumentors
    "BaseSpanInstrumentor",
    "GuardrailSpanInstrumentor",
    "InstrumentationState",
    "LlmSpanInstrumentor",
    "ToolSpanInstrumentor",
    # Span attributes
    "AgentRunSpanAttributes",
    "AgentToolSpanAttributes",
    "CompletionSpanAttributes",
    "ContextGroundingToolSpanAttributes",
    "ErrorDetails",
    "EscalationToolSpanAttributes",
    "ExecutionType",
    "GovernanceSpanAttributes",
    "GuardrailEscalationSpanAttributes",
    "GuardrailEvaluationSpanAttributes",
    "IntegrationToolSpanAttributes",
    "LlmCallSpanAttributes",
    "McpSessionStartSpanAttributes",
    "McpToolSpanAttributes",
    "ProcessToolSpanAttributes",
    "SpanType",
    "ToolCallSpanAttributes",
    # Infrastructure
    "SpanHierarchyManager",
    "SpanKeys",
    # Span filters
    "is_azure_monitor_span",
    "is_custom_instrumentation_span",
    "is_http_instrumentation_span",
    "is_openinference_span",
    # Trace context storage
    "PendingSpanData",
    "SqliteTraceContextStorage",
    "TraceContextData",
    "TraceContextStorage",
    # Context variables
    "reference_id_context",
]
