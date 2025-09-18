import json
import logging
from typing import Any, Dict

from opentelemetry.sdk.trace.export import (
    SpanExportResult,
)
from uipath.tracing import LlmOpsHttpExporter

from ._utils import _setup_tracer_httpx_logging

logger = logging.getLogger(__name__)

_setup_tracer_httpx_logging("/llmops_/api/Agent/trace/")


class LangchainExporter(LlmOpsHttpExporter):
    # Mapping of old attribute names to new attribute names or (new name, function)
    ATTRIBUTE_MAPPING = {
        "input.value": ("input", lambda s: json.loads(s)),
        "output.value": ("output", lambda s: json.loads(s)),
        "llm.model_name": "model",
    }

    # Mapping of span types
    SPAN_TYPE_MAPPING = {
        "LLM": "completion",
        # Add more mappings as needed
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _send_with_retries(
        self, url: str, payload: list[Dict[str, Any]], max_retries: int = 4
    ) -> SpanExportResult:
        # Transform attributes in each span's payload before sending
        for span_data in payload:
            if "Attributes" in span_data and isinstance(span_data["Attributes"], str):
                try:
                    # Parse the JSON string to a dictionary
                    attributes = json.loads(span_data["Attributes"])

                    if "openinference.span.kind" in attributes:
                        span_type = attributes["openinference.span.kind"]
                        # Map span type using SPAN_TYPE_MAPPING
                        span_data["SpanType"] = self.SPAN_TYPE_MAPPING.get(
                            span_type, span_type
                        )

                    # Apply the transformation logic
                    for old_key, mapping in self.ATTRIBUTE_MAPPING.items():
                        if old_key in attributes:
                            if isinstance(mapping, tuple):
                                new_key, func = mapping
                                try:
                                    attributes[new_key] = func(attributes[old_key])
                                except Exception:
                                    attributes[new_key] = attributes[old_key]
                            else:
                                new_key = mapping
                                attributes[new_key] = attributes[old_key]

                    # Transform token usage data if present
                    token_keys = {
                        "llm.token_count.prompt": "promptTokens",
                        "llm.token_count.completion": "completionTokens",
                        "llm.token_count.total": "totalTokens",
                    }

                    # Check if any token count keys exist
                    if any(key in attributes for key in token_keys):
                        usage = {}
                        for old_key, new_key in token_keys.items():
                            if old_key in attributes:
                                usage[new_key] = attributes[old_key]
                                del attributes[old_key]

                        # Add usage to attributes
                        attributes["usage"] = usage

                    # Convert back to JSON string
                    span_data["attributes"] = json.dumps(attributes)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse attributes JSON: {e}")

        return super()._send_with_retries(
            url=url,
            payload=payload,
            max_retries=max_retries,
        )
