"""Span exporter wrapper for redacting PII before export.

Environment variables:
    DISABLE_OTEL_MASKING: Set to "true" to disable PII redaction (default: false)
"""

import fnmatch
import json
import logging
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)


def _get_pii_attributes() -> frozenset[str]:
    """Get the set of attributes to redact (exact match)."""
    # Default attributes that commonly contain PII in LangGraph/OpenInference spans
    return frozenset(
        {
            "input.value",  # Full input messages/prompts
            "output.value",  # Full output/responses
        }
    )


def _get_pii_patterns() -> list[str]:
    """Get the list of glob patterns for attributes to redact."""
    # Default patterns for dynamic attribute names
    return [
        "llm.*.message.content",  # e.g., llm.input_messages.0.message.content
        "llm.*.message_content",  # e.g., llm.output_messages.1.message_content
        "*.messages.*.content",  # Generic message content patterns
        "llm.*.message_content.text",
        "llm.*.function.arguments",
        "*.invocation_parameters",
        "*.json_schema",
    ]


def _get_preserve_fields() -> frozenset[str]:
    """Get the set of JSON fields to preserve."""
    # Default fields to preserve (commonly non-PII)
    return frozenset({"inner_state", "job_attachments", "graph", "goto", "type"})


def _get_redactable_exception_types() -> frozenset[str]:
    """Get exception type substrings that should trigger redaction."""
    return frozenset(
        {
            "langgraph.errors.GraphInterrupt",
        }
    )


_PII_ATTRIBUTES = _get_pii_attributes()
_PII_PATTERNS = _get_pii_patterns()
_PRESERVE_FIELDS = _get_preserve_fields()
_REDACTABLE_EXCEPTION_TYPES = _get_redactable_exception_types()


def _redact_dict_selectively(data: dict[str, Any]) -> dict[str, Any]:
    """Redact all fields except those in the preserve list.

    Args:
        data: Dictionary to selectively redact

    Returns:
        New dictionary with fields redacted except preserved ones
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in _PRESERVE_FIELDS:
            # Preserve this field completely, including all nested content
            result[key] = value
        else:
            # Redact this field - return string representation
            redacted_value: str
            if isinstance(value, str):
                redacted_value = f"[REDACTED: string, length={len(value)}]"
            elif isinstance(value, list):
                redacted_value = f"[REDACTED: list, length={len(value)}]"
            elif isinstance(value, dict):
                redacted_value = f"[REDACTED: dict, keys={len(value)}]"
            else:
                redacted_value = f"[REDACTED: {type(value).__name__}]"
            result[key] = redacted_value

    return result


def _redact_list_selectively(data: list[Any]) -> list[Any]:
    """Redact specific fields in list items while preserving others.

    Args:
        data: List to selectively redact

    Returns:
        New list with sensitive fields redacted
    """
    result: list[Any] = []
    for item in data:
        if isinstance(item, dict):
            result.append(_redact_dict_selectively(item))
        elif isinstance(item, list):
            result.append(_redact_list_selectively(item))
        else:
            # Primitive types in lists are kept as-is for now
            # Could be made more strict if needed
            result.append(item)
    return result


def _redact_value(value: Any) -> str | dict[str, Any]:
    """Redact a value while preserving type information.

    If preserve_fields is configured and the value is JSON/dict, will preserve
    only the specified fields and redact all others.

    Args:
        value: The value to redact (can be string, dict, list, etc.)

    Returns:
        A redacted string or dict (if selective redaction is applied)
    """
    if isinstance(value, str):
        try:
            # Try to parse as JSON to detect structured data
            parsed = json.loads(value)
            if _PRESERVE_FIELDS and isinstance(parsed, dict):
                # Return selectively redacted dict as JSON string
                redacted = _redact_dict_selectively(parsed)
                return json.dumps(redacted)
            else:
                # Fall back to full redaction
                return _redact_value(parsed)
        except (json.JSONDecodeError, TypeError):
            # Plain string - redact entirely
            return f"[REDACTED: string, length={len(value)}]"
    elif isinstance(value, dict):
        if _PRESERVE_FIELDS:
            # Return selectively redacted dict
            return _redact_dict_selectively(value)
        else:
            # Full redaction
            num_keys = len(value)
            keys = list(value.keys())[:5]  # Show first 5 keys for context
            return f"[REDACTED: dict, keys={num_keys}, sample_keys={keys}]"
    elif isinstance(value, list):
        # Redact list - show count but redact items
        return f"[REDACTED: list, length={len(value)}]"
    else:
        return f"[REDACTED: {type(value).__name__}]"


def _should_redact_attribute(attr_name: str) -> bool:
    """Check if an attribute should be redacted based on exact match or pattern.

    Args:
        attr_name: The attribute name to check

    Returns:
        True if the attribute should be redacted
    """
    # Check exact match
    if attr_name in _PII_ATTRIBUTES:
        return True

    # Check pattern match using glob patterns
    for pattern in _PII_PATTERNS:
        if fnmatch.fnmatch(attr_name, pattern):
            logger.debug(
                "Attribute '%s' matches pattern '%s' - will redact", attr_name, pattern
            )
            return True

    return False


def _redact_attributes(attributes: Attributes | None) -> dict[str, Any]:
    """Redact PII from span attributes based on configured patterns.

    Supports both exact attribute name matching and glob pattern matching:
    - Exact: "input.value", "output.value"
    - Pattern: "llm.*.message.content" matches "llm.input_messages.0.message.content"

    Args:
        attributes: Original span attributes

    Returns:
        New attributes dict with matched attributes redacted
    """
    if not attributes:
        return {}

    redacted = dict(attributes)

    # Check all attributes for redaction (exact match or pattern match)
    for attr_key in list(redacted.keys()):
        if _should_redact_attribute(attr_key):
            original_value = redacted[attr_key]
            redacted_value = _redact_value(original_value)

            # If selective redaction returns a dict, convert to JSON string
            if isinstance(redacted_value, dict):
                redacted[attr_key] = json.dumps(redacted_value)
            else:
                redacted[attr_key] = redacted_value

            logger.debug(
                "Redacted attribute: %s (preserve_fields=%s)",
                attr_key,
                bool(_PRESERVE_FIELDS),
            )

    return redacted


def _should_redact_exception(exception_type: Any) -> bool:
    if not exception_type or not isinstance(exception_type, str):
        return False

    exception_type_lower = exception_type.lower()
    for redactable_type in _REDACTABLE_EXCEPTION_TYPES:
        if redactable_type.lower() in exception_type_lower:
            return True

    return False


def _redact_events(events: Sequence[Event] | None) -> list[Event]:
    """Redact PII from span events for specific exception types."""
    if not events:
        return []

    redacted_events = []
    for event in events:
        if not event.attributes:
            redacted_events.append(event)
            continue

        event_attrs = dict(event.attributes)
        exception_type = event_attrs.get("exception.type")
        should_redact = _should_redact_exception(exception_type)

        if should_redact:
            if "exception.message" in event_attrs:
                original_msg = event_attrs["exception.message"]
                if isinstance(original_msg, str):
                    event_attrs["exception.message"] = (
                        f"[REDACTED: exception message, length={len(original_msg)}]"
                    )

            if "exception.stacktrace" in event_attrs:
                original_trace = event_attrs["exception.stacktrace"]
                if isinstance(original_trace, str):
                    line_count = original_trace.count("\n") + 1
                    event_attrs["exception.stacktrace"] = (
                        f"[REDACTED: stack trace, lines={line_count}]"
                    )

        redacted_event = Event(
            name=event.name, attributes=event_attrs, timestamp=event.timestamp
        )
        redacted_events.append(redacted_event)

    return redacted_events


class _RedactedSpan(ReadableSpan):
    """Wrapper around ReadableSpan that provides redacted attributes and events."""

    def __init__(
        self,
        span: ReadableSpan,
        redacted_attributes: Attributes,
        redacted_events: Sequence[Event],
    ):
        """Initialize with original span and redacted data.

        Args:
            span: Original span to wrap
            redacted_attributes: Redacted version of attributes
            redacted_events: Redacted version of events
        """
        self._span = span
        self._redacted_attributes = redacted_attributes
        self._redacted_events = redacted_events

    @property
    def name(self) -> str:
        return self._span.name

    @property
    def context(self):
        return self._span.context

    @property
    def parent(self):
        return self._span.parent

    @property
    def start_time(self) -> int:
        return self._span.start_time  # type: ignore[return-value]

    @property
    def end_time(self) -> int:
        return self._span.end_time  # type: ignore[return-value]

    @property
    def status(self):
        return self._span.status

    @property
    def attributes(self) -> Attributes:
        """Return redacted attributes instead of original."""
        return self._redacted_attributes

    @property
    def events(self) -> Sequence[Event]:
        """Return redacted events instead of original."""
        return self._redacted_events

    @property
    def links(self):
        return self._span.links

    @property
    def kind(self):
        return self._span.kind

    @property
    def resource(self):
        return self._span.resource

    @property
    def instrumentation_scope(self):
        return self._span.instrumentation_scope

    @property
    def instrumentation_info(self):
        return self._span.instrumentation_info


class PIIFilteringExporter(SpanExporter):
    """Wraps a SpanExporter to redact PII from attributes and events before export.

    This exporter intercepts spans and removes/redacts sensitive data:
    - Attributes that match configured patterns or exact names (e.g., input.value,
      output.value, llm.*.message.content) which commonly contain PII
    - Exception information in events for specific exception types

    The exporter supports two redaction modes:

    1. Full redaction (default - when no preserve fields are configured):
       - Replaces entire value with type/size metadata
       - Example: input.value becomes "[REDACTED: dict, keys=2, sample_keys=['messages']]"

    2. Selective redaction (when preserve fields are configured):
       - Redacts all fields EXCEPT those in the preserve list
       - Example with preserve fields ["inner_state", "job_attachments"]:
         {"messages": [...], "inner_state": {...}}
         becomes: {"messages": "[REDACTED: list, length=2]", "inner_state": {...}}

    The exporter:
    - Only redacts attributes matching configured patterns or exact attribute names
    - Only redacts exceptions whose type contains configured substrings
      (e.g., "langgraph.errors.GraphInterrupt")
    - Redacts exception.message and exception.stacktrace for matching exception types
    - Preserves exception.type for debugging
    - Preserves span structure, IDs, and timing for correlation

    Usage:
        azure_exporter = AzureMonitorTraceExporter(...)
        filtered_exporter = PIIFilteringExporter(azure_exporter)
        trace_manager.add_span_exporter(filtered_exporter)
    """

    def __init__(self, delegate: SpanExporter):
        """Initialize the PII filtering exporter.

        Args:
            delegate: The underlying exporter to send filtered spans to.
        """
        self._delegate = delegate

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans with PII redacted from attributes and events."""
        redacted_spans = []
        for span in spans:
            redacted_attrs = _redact_attributes(span.attributes)
            redacted_events = _redact_events(span.events)
            redacted_span = _RedactedSpan(span, redacted_attrs, redacted_events)
            redacted_spans.append(redacted_span)

        return self._delegate.export(redacted_spans)

    def shutdown(self) -> None:
        """Shutdown the delegate exporter."""
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the delegate exporter."""
        return self._delegate.force_flush(timeout_millis)
