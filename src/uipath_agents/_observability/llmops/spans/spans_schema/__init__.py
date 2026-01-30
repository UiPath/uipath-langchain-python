"""Span schemas for LLMOps traces.

Modular span creation organized by span type.
"""

from .agent import AgentSpanSchema
from .base import (
    SpanUpsertProtocol,
    SyntheticReadableSpan,
    apply_attributes,
    create_span,
    end_span_error,
    end_span_ok,
    get_parent_context,
    reference_id_context,
)
from .guardrails import GuardrailSpanSchema
from .llm import LlmSpanSchema
from .tool import ToolSpanSchema

__all__ = [
    # Base utilities
    "SyntheticReadableSpan",
    "SpanUpsertProtocol",
    "reference_id_context",
    "apply_attributes",
    "get_parent_context",
    "create_span",
    "end_span_ok",
    "end_span_error",
    # Span schemas
    "AgentSpanSchema",
    "LlmSpanSchema",
    "ToolSpanSchema",
    "GuardrailSpanSchema",
]
