import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, MutableMapping, override

logger = logging.getLogger(__name__)


def try_convert_json(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tries to convert stringified JSON values in a flattened dictionary back to their original types.

    Args:
        flat_dict: A dictionary with potentially stringified JSON values.

    Returns:
        A new dictionary with JSON strings converted to their original types.
    """
    result = {}
    for key, value in flat_dict.items():
        if isinstance(value, str):
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value
        else:
            result[key] = value
    return result


def unflatten_dict(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a flattened dictionary with dot-separated keys into a nested dictionary.

    Args:
        flat_dict: Dictionary with dot-separated keys (e.g., 'llm.output_messages.0.message.content')

    Returns:
        Nested dictionary structure

    Example:
        Input: {'llm.output_messages.0.message.content': 'hello', 'llm.model': 'gpt-4'}
        Output: {'llm': {'output_messages': [{'message': {'content': 'hello'}}], 'model': 'gpt-4'}}
    """
    result = {}

    for key, value in flat_dict.items():
        # Split the key by dots
        parts = key.split(".")
        current = result

        # Navigate/create the nested structure
        for i, part in enumerate(parts[:-1]):
            # Check if this part represents an array index
            if part.isdigit():
                # Convert to integer index
                index = int(part)
                # Ensure the parent is a list
                if not isinstance(current, list):
                    raise ValueError(
                        f"Expected list but found {type(current)} for key: {key}"
                    )
                # Extend the list if necessary
                while len(current) <= index:
                    current.append(None)

                # If the current element is None, we need to create a structure for it
                if current[index] is None:
                    # Look ahead to see if the next part is a digit (array index)
                    next_part = parts[i + 1] if i + 1 < len(parts) else None
                    if next_part and next_part.isdigit():
                        current[index] = []
                    else:
                        current[index] = {}

                current = current[index]
            else:
                # Regular dictionary key
                if part not in current:
                    # Look ahead to see if the next part is a digit (array index)
                    next_part = parts[i + 1] if i + 1 < len(parts) else None
                    if next_part and next_part.isdigit():
                        current[part] = []
                    else:
                        current[part] = {}
                current = current[part]  # Set the final value

        final_key = parts[-1]
        if final_key.isdigit():
            # If the final key is a digit, we're setting an array element
            index = int(final_key)
            if not isinstance(current, list):
                raise ValueError(
                    f"Expected list but found {type(current)} for key: {key}"
                )
            while len(current) <= index:
                current.append(None)
            current[index] = value
        else:
            # Regular key assignment
            current[final_key] = value

    return result


def safe_get(data: Dict[str, Any], path: str, default=None):
    """Safely get nested value using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def safe_parse_json(value):
    """Safely parse JSON string."""
    if isinstance(value, str):
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return value
    return value


class BaseSpanProcessor(ABC):
    """
    Abstract base class for span processors.

    Defines the interface for processing spans with a single abstract method.
    """

    @abstractmethod
    def process_span(self, span_data: MutableMapping[str, Any]) -> Dict[str, Any]:
        """
        Process a span and return the transformed data.

        Args:
            span_data: The span data to process

        Returns:
            Processed span data
        """
        pass


class LangchainSpanProcessor(BaseSpanProcessor):
    """
    A class to process spans, applying custom attribute and type mappings.

    This processor can transform flattened attribute keys (e.g., 'llm.output_messages.0.message.role')
    into nested dictionary structures for easier access and processing.

    Example usage:
        # With unflattening enabled
        processor = LangchainSpanProcessor(unflatten_attributes=True, dump_attributes_as_string=False)
        processed_span = processor.process_span(span_data)

        # Access nested attributes naturally:
        role = processed_span['attributes']['llm']['output_messages'][0]['message']['role']

        # Without unflattening (original behavior)
        processor = LangchainSpanProcessor(unflatten_attributes=False)
        processed_span = processor.process_span(span_data)

        # Access with flattened keys:
        role = processed_span['attributes']['llm.output_messages.0.message.role']
    """

    # Mapping of old attribute names to new attribute names or (new name, function)
    ATTRIBUTE_MAPPING = {
        "input.value": ("input", lambda s: json.loads(s)),
        "output.value": ("output", lambda s: json.loads(s)),
        "llm.model_name": "model",
    }

    # Mapping of span types
    SPAN_TYPE_MAPPING = {
        "LLM": "completion",
        "TOOL": "toolCall",
        # Add more mappings as needed
    }

    def __init__(
        self,
        dump_attributes_as_string: bool = True,
        unflatten_attributes: bool = True,
        map_json_fields: bool = True,
    ):
        """
        Initializes the LangchainSpanProcessor.

        Args:
            dump_attributes_as_string: If True, dumps attributes as a JSON string.
                                       Otherwise, attributes are set as a dictionary.
            unflatten_attributes: If True, converts flattened dot-separated keys
                                  into nested dictionary structures.
            map_json_fields: If True, applies JSON field mapping transformations
                            for tool calls and LLM calls.
        """
        self._dump_attributes_as_string = dump_attributes_as_string
        self._unflatten_attributes = unflatten_attributes
        self._map_json_fields = map_json_fields

    def extract_attributes(self, span_data: MutableMapping[str, Any]) -> Dict[str, Any]:
        """Extract and parse attributes from span_data, checking both 'Attributes' and 'attributes' keys."""
        for key in ["Attributes", "attributes"]:
            if key in span_data:
                value = span_data.pop(key)
                return safe_parse_json(value) if isinstance(value, str) else value
        return None

    @override
    def process_span(self, span_data: MutableMapping[str, Any]) -> Dict[str, Any]:
        attributes = self.extract_attributes(span_data)

        if attributes:
            if "openinference.span.kind" in attributes:
                # Remove the span kind attribute
                span_type = attributes["openinference.span.kind"]
                # Map span type using SPAN_TYPE_MAPPING
                span_data["SpanType"] = self.SPAN_TYPE_MAPPING.get(span_type, span_type)
                del attributes["openinference.span.kind"]

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
                    del attributes[old_key]

        if attributes:
            # Apply unflattening if requested (before JSON field mapping)
            if self._unflatten_attributes:
                try:
                    attributes = try_convert_json(attributes)
                    attributes = unflatten_dict(attributes)
                except Exception as e:
                    logger.warning(f"Failed to unflatten attributes: {e}")

            # Set attributes in span_data as dictionary for JSON field mapping
            span_data["attributes"] = attributes

            # Apply JSON field mapping before final serialization
            if self._map_json_fields:
                span_data = self.map_json_fields_from_attributes(span_data)

            # Convert back to JSON string if requested (after all transformations)
            if self._dump_attributes_as_string:
                span_data["attributes"] = json.dumps(span_data["attributes"])

        return span_data

    def map_tool_call_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Simple tool call mapping - just add new fields."""
        result = attributes.copy()  # Keep originals

        # Add new fields
        result["type"] = "toolCall"
        result["callId"] = attributes.get("call_id") or attributes.get("id")
        result["toolName"] = safe_get(attributes, "tool.name")
        result["arguments"] = safe_parse_json(attributes.get("input", "{}"))
        result["toolType"] = "Integration"
        result["result"] = safe_parse_json(attributes.get("output"))
        result["error"] = None

        return result

    def map_llm_call_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Simple LLM call mapping - just add new fields."""
        result = attributes.copy()  # Keep originals

        # Transform token usage data if present (after unflattening)
        # Use safe_get to extract token count values from nested structure
        prompt_tokens = safe_get(attributes, "llm.token_count.prompt")
        completion_tokens = safe_get(attributes, "llm.token_count.completion")
        total_tokens = safe_get(attributes, "llm.token_count.total")

        usage = {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
            "isByoExecution": False,
            "executionDeploymentType": "PAYGO",
            "isPiiMasked": False,
        }

        # remove None values
        usage = {k: v for k, v in usage.items() if v is not None}

        result["usage"] = usage

        # Add new fields
        result["input"] = safe_get(attributes, "llm.input_messages")
        result["output"] = safe_get(attributes, "llm.output_messages")

        result["type"] = "completion"
        result["model"] = safe_get(attributes, "llm.invocation_parameters.model")

        # Settings
        settings = {}
        max_tokens = safe_get(attributes, "llm.invocation_parameters.max_tokens")
        temperature = safe_get(attributes, "llm.invocation_parameters.temperature")
        if max_tokens:
            settings["maxTokens"] = max_tokens
        if temperature is not None:
            settings["temperature"] = temperature
        if settings:
            result["settings"] = settings

        # Tool calls (simplified)
        tool_calls = []
        output_msgs = safe_get(attributes, "llm.output_messages", [])
        for msg in output_msgs:
            msg_tool_calls = safe_get(msg, "message.tool_calls", [])
            for tc in msg_tool_calls:
                tool_call_data = tc.get("tool_call", {})
                tool_calls.append(
                    {
                        "id": tool_call_data.get("id"),
                        "name": safe_get(tool_call_data, "function.name"),
                        "arguments": safe_get(tool_call_data, "function.arguments", {}),
                    }
                )
        if tool_calls:
            result["toolCalls"] = tool_calls

        # Usage (enhance existing if not created above)
        if "usage" in result:
            usage = result["usage"]
            if isinstance(usage, dict):
                usage.setdefault("isByoExecution", False)
                usage.setdefault("executionDeploymentType", "PAYGO")
                usage.setdefault("isPiiMasked", False)

        return result

    def map_json_fields_from_attributes(
        self, span_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simple mapping dispatcher."""
        if "attributes" not in span_data:
            return span_data

        attributes = span_data["attributes"]

        # Parse if string
        if isinstance(attributes, str):
            try:
                attributes = json.loads(attributes)
            except json.JSONDecodeError:
                return span_data

        if not isinstance(attributes, dict):
            return span_data

        # Simple detection and mapping
        if "tool" in attributes or span_data.get("SpanType") == "toolCall":
            span_data["attributes"] = self.map_tool_call_attributes(attributes)
        elif "llm" in attributes or span_data.get("SpanType") == "completion":
            span_data["attributes"] = self.map_llm_call_attributes(attributes)

        return span_data
