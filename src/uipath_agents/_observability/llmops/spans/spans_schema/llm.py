"""LLM span schemas.

Handles LLM call and model run spans.
"""

import logging
import uuid
from contextvars import Token
from typing import Callable, Optional, Tuple

from opentelemetry.trace import (
    Span,
    SpanKind,
    Tracer,
)

from ..span_attributes import (
    CompletionSpanAttributes,
    LlmCallSpanAttributes,
    ModelSettings,
)
from ..span_name import SpanName
from .base import apply_attributes, create_span, license_ref_id_context

logger = logging.getLogger(__name__)

__all__ = [
    "LlmSpanSchema",
]


class LlmSpanSchema:
    """Schema for LLM-related spans."""

    def __init__(
        self,
        tracer: Tracer,
        upsert_started_fn: Optional[Callable[[Span], bool]] = None,
    ):
        """Initialize LLM span schema.

        Args:
            tracer: The OpenTelemetry tracer to use
            upsert_started_fn: Optional function to upsert span on start
        """
        self._tracer = tracer
        self._upsert_started = upsert_started_fn

    def start_llm_call(
        self,
        input: Optional[str] = None,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start an LLM call span (outer wrapper for LLM iteration).

        Returns a span that must be explicitly ended.
        Use for callback-based instrumentation where context managers don't fit.

        Args:
            input: The user input/prompt for this LLM call
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            The started Span (caller must call span.end())
        """
        span = create_span(
            self._tracer,
            SpanName.LLM_CALL,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )
        attrs = LlmCallSpanAttributes(input=input)
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span

    def start_model_run(
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        parent_span: Optional[Span] = None,
    ) -> Tuple[Span, Token[Optional[str]]]:
        """Start a model run span (inner actual API call).

        Should be a child of an LLM call span.

        Args:
            model_name: Name of the model being called
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            parent_span: Optional parent span. If None, uses current span.

        Returns:
            Tuple of (started Span, ContextVar token for license_ref_id).
            Caller must call span.end() and reset the token via
            license_ref_id_context.reset(token).
        """
        span = create_span(
            self._tracer,
            SpanName.MODEL_RUN,
            parent_span=parent_span,
            kind=SpanKind.INTERNAL,
        )

        license_ref_id = str(uuid.uuid4())
        license_token = license_ref_id_context.set(license_ref_id)
        logger.info("start_model_run: generated licenseRefId=%s", license_ref_id)

        # Model run: type="completion", has model and nested settings
        settings = None
        if max_tokens is not None or temperature is not None:
            settings = ModelSettings(max_tokens=max_tokens, temperature=temperature)
        attrs = CompletionSpanAttributes(
            model=model_name,
            settings=settings,
            license_ref_id=license_ref_id,
        )
        apply_attributes(span, attrs)
        if self._upsert_started:
            self._upsert_started(span)
        return span, license_token
