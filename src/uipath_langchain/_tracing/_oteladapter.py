import json
import logging
from typing import Any, Dict, List

from opentelemetry.sdk.trace.export import (
    SpanExportResult,
)
from uipath.tracing import LlmOpsHttpExporter

logger = logging.getLogger(__name__)


def _safe_parse_json(s: Any) -> Any:
    """Safely parse a JSON string, returning the original if not a string or on error."""
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def _get_llm_messages(attributes: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    """Extracts and reconstructs LLM messages from flattened attributes."""
    messages: dict[int, dict[str, Any]] = {}
    message_prefix = f"{prefix}."

    for key, value in attributes.items():
        if key.startswith(message_prefix):
            parts = key[len(message_prefix) :].split(".")
            if len(parts) >= 2 and parts[0].isdigit():
                index = int(parts[0])
                if index not in messages:
                    messages[index] = {}
                current: Any = messages[index]

                for i, part in enumerate(parts[1:-1]):
                    key_part: str | int = part
                    if part.isdigit() and (
                        i + 2 < len(parts) and parts[i + 2].isdigit()
                    ):
                        key_part = int(part)

                    if isinstance(current, dict):
                        if key_part not in current:
                            current[key_part] = {}
                        current = current[key_part]
                    elif isinstance(current, list) and isinstance(key_part, int):
                        if key_part >= len(current):
                            current.append({})
                        current = current[key_part]

                current[parts[-1]] = value

    # Convert dict to list, ordered by index
    return [messages[i] for i in sorted(messages.keys())]


class LangChainExporter(LlmOpsHttpExporter):
    # Mapping of old attribute names to new attribute names or (new name, function)
    ATTRIBUTE_MAPPING: dict[str, str | tuple[str, Any]] = {
        "input.value": ("input", _safe_parse_json),
        "output.value": ("output", _safe_parse_json),
        "llm.model_name": "model",
    }

    # Mapping of span types
    SPAN_TYPE_MAPPING: dict[str, str] = {
        "LLM": "completion",
        "TOOL": "toolCall",
        # Add more mappings as needed
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _map_llm_call_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Maps attributes for LLM calls, handling flattened keys."""
        result = attributes.copy()  # Keep original attributes including basic mappings

        # Token Usage
        token_keys = {
            "llm.token_count.prompt": "promptTokens",
            "llm.token_count.completion": "completionTokens",
            "llm.token_count.total": "totalTokens",
        }
        usage = {
            new_key: attributes.get(old_key)
            for old_key, new_key in token_keys.items()
            if old_key in attributes
        }
        if usage:
            result["usage"] = usage

        # Input/Output Messages
        result["input"] = _get_llm_messages(attributes, "llm.input_messages")
        output_messages = _get_llm_messages(attributes, "llm.output_messages")
        result["output"] = output_messages

        # Invocation Parameters
        invocation_params = _safe_parse_json(
            attributes.get("llm.invocation_parameters", "{}")
        )
        if isinstance(invocation_params, dict):
            result["model"] = invocation_params.get("model", result.get("model"))
            settings: dict[str, Any] = {}
            if "max_tokens" in invocation_params:
                settings["maxTokens"] = invocation_params["max_tokens"]
            if "temperature" in invocation_params:
                settings["temperature"] = invocation_params["temperature"]
            if settings:
                result["settings"] = settings

        # Tool Calls
        tool_calls: list[dict[str, Any]] = []
        for msg in output_messages:
            # Ensure msg is a dictionary before proceeding
            if not isinstance(msg, dict):
                continue
            msg_tool_calls = msg.get("message", {}).get("tool_calls", [])

            # Ensure msg_tool_calls is a list
            if not isinstance(msg_tool_calls, list):
                continue

            for tc in msg_tool_calls:
                if not isinstance(tc, dict):
                    continue
                tool_call_data = tc.get("tool_call", {})
                if not isinstance(tool_call_data, dict):
                    continue
                tool_calls.append(
                    {
                        "id": tool_call_data.get("id"),
                        "name": tool_call_data.get("function", {}).get("name"),
                        "arguments": _safe_parse_json(
                            tool_call_data.get("function", {}).get("arguments", "{}")
                        ),
                    }
                )
        if tool_calls:
            result["toolCalls"] = tool_calls

        return result

    def _map_tool_call_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Maps attributes for tool calls."""
        result = attributes.copy()  # Keep original attributes

        result["type"] = "toolCall"
        result["callId"] = attributes.get("call_id") or attributes.get("id")
        result["toolName"] = attributes.get("tool.name")
        result["arguments"] = _safe_parse_json(
            attributes.get("input", attributes.get("input.value", "{}"))
        )
        result["toolType"] = "Integration"
        result["result"] = _safe_parse_json(
            attributes.get("output", attributes.get("output.value"))
        )
        result["error"] = None

        return result

    def _process_span_attributes(self, span_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts, transforms, and maps attributes for a span."""
        if "Attributes" not in span_data:
            return span_data

        logger.debug(f"Processing span: {span_data}")

        attributes_val = span_data["Attributes"]
        if isinstance(attributes_val, str):
            try:
                attributes: Dict[str, Any] = json.loads(attributes_val)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse attributes JSON: {e}")
                return span_data
        elif isinstance(attributes_val, dict):
            attributes = attributes_val
        else:
            return span_data

        # Determine SpanType
        if "openinference.span.kind" in attributes:
            span_type = attributes["openinference.span.kind"]
            span_data["SpanType"] = self.SPAN_TYPE_MAPPING.get(span_type, span_type)

        # Apply basic attribute mapping
        for old_key, mapping in self.ATTRIBUTE_MAPPING.items():
            if old_key in attributes:
                if isinstance(mapping, tuple):
                    new_key, func = mapping
                    attributes[new_key] = func(attributes[old_key])
                else:
                    new_key = mapping
                    attributes[new_key] = attributes[old_key]

        # Apply detailed mapping based on SpanType
        span_type = span_data.get("SpanType")
        if span_type == "completion":
            processed_attributes = self._map_llm_call_attributes(attributes)
        elif span_type == "toolCall":
            processed_attributes = self._map_tool_call_attributes(attributes)
        else:
            processed_attributes = attributes.copy()

        span_data["Attributes"] = json.dumps(processed_attributes)

        logger.info(f"Transformed span: {span_data}")
        return span_data

    def _send_with_retries(
        self, url: str, payload: List[Dict[str, Any]], max_retries: int = 4
    ) -> SpanExportResult:
        # Transform attributes in each span's payload before sending
        transformed_payload = [self._process_span_attributes(span) for span in payload]

        return super()._send_with_retries(
            url=url,
            payload=transformed_payload,
            max_retries=max_retries,
        )
