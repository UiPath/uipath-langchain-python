"""Attribute extraction and formatting helpers for span instrumentation."""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.trace import Span
from pydantic import BaseModel
from uipath.core.serialization import serialize_json
from uipath.platform.resume_triggers import is_no_content_marker
from uipath.tracing import AttachmentDirection, AttachmentProvider, SpanAttachment

from uipath_agents._observability.llmops.spans.span_attributes import Usage

logger = logging.getLogger(__name__)

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\-_]+=*$")


def sanitize_file_data(obj: Any) -> Any:
    """Recursively sanitize content, replacing file data with placeholders."""
    if isinstance(obj, bytes):
        return f"<bytes: {len(obj)} bytes>"
    if isinstance(obj, str):
        if obj.startswith("data:") and ";base64," in obj:
            return "<base64 data omitted>"
        if len(obj) > 1000 and _BASE64_RE.match(obj):
            return "<base64 data omitted>"
        return obj
    if isinstance(obj, list):
        return [sanitize_file_data(item) for item in obj]
    if isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            if key in ("data", "bytes", "file_data", "image_url") and not isinstance(
                value, dict
            ):
                if isinstance(value, bytes):
                    sanitized[key] = f"<bytes: {len(value)} bytes>"
                elif (
                    isinstance(value, str)
                    and len(value) > 1000
                    and _BASE64_RE.match(value)
                ):
                    sanitized[key] = "<base64 data omitted>"
                elif isinstance(value, list):
                    sanitized[key] = sanitize_file_data(value)
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = sanitize_file_data(value)
        return sanitized
    return obj


def extract_model_name(serialized: Dict[str, Any]) -> str:
    """Extract actual model name from LLM serialized data.

    Checks multiple locations where LangChain stores model info:
    - kwargs.model_name, kwargs.model (most common)
    - kwargs.model_id (some providers)
    - kwargs.deployment_name, kwargs.azure_deployment (Azure OpenAI)
    - serialized.id[-1] only if it looks like a model name (not a class)
    """
    kwargs = serialized.get("kwargs", {})

    model = (
        kwargs.get("model_name")
        or kwargs.get("model")
        or kwargs.get("model_id")
        or kwargs.get("deployment_name")
        or kwargs.get("azure_deployment")
    )
    if model:
        return model

    # Check serialized.id - but only use if it looks like a model name
    id_list = serialized.get("id", [])
    if id_list:
        last_id = id_list[-1] if isinstance(id_list, list) else str(id_list)
        model_indicators = [
            "gpt",
            "claude",
            "gemini",
            "llama",
            "mistral",
            "4o",
            "3.5",
            "4-",
            "/",
        ]
        if any(indicator in str(last_id).lower() for indicator in model_indicators):
            return str(last_id)

    return serialized.get("name", "unknown")


def extract_settings(
    serialized: Dict[str, Any],
) -> Tuple[Optional[int], Optional[float]]:
    """Extract max_tokens and temperature from LLM serialized data."""
    kwargs = serialized.get("kwargs", {})
    max_tokens = kwargs.get("max_tokens")
    temperature = kwargs.get("temperature")
    return max_tokens, temperature


def extract_headers(response: LLMResult) -> dict[str, Any]:
    """Extract headers from LLM response."""
    if response.generations and response.generations[0]:
        gen = response.generations[0][0]
        if not isinstance(gen, ChatGeneration):
            return {}

        response_metadata = gen.message.response_metadata
        if "headers" in response_metadata:
            headers = response_metadata.get("headers")
        else:
            headers = response_metadata.get("ResponseMetadata", {}).get("HTTPHeaders")
    return headers or {}


def set_usage_attributes(span: Span, response: Optional[LLMResult]) -> None:
    """Set token usage attributes on span from LLM response."""
    if not response:
        return

    token_usage = None

    # Modern path: message.usage_metadata (LangChain 0.2+)
    if response.generations and response.generations[0]:
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
            um = msg.usage_metadata
            token_usage = {
                "prompt_tokens": um.get("input_tokens", 0),
                "completion_tokens": um.get("output_tokens", 0),
                "total_tokens": um.get("total_tokens", 0),
            }

    # Legacy fallback: llm_output
    if not token_usage and response.llm_output:
        token_usage = response.llm_output.get("token_usage") or response.llm_output.get(
            "usage"
        )

    token_usage = token_usage or {}
    headers = extract_headers(response)

    if not token_usage and not headers:
        return

    usage = Usage(
        llm_calls=1,
        completion_tokens=token_usage.get("completion_tokens", 0),
        prompt_tokens=token_usage.get("prompt_tokens", 0),
        total_tokens=token_usage.get("total_tokens", 0),
        is_byo_execution=bool(headers.get("x-uipath-llmgateway-isbyoexecution", False)),
        execution_deployment_type=headers.get(
            "x-uipath-llmgateway-executiondeploymenttype"
        ),
        is_pii_masked=bool(headers.get("x-uipath-llmgateway-ispiimasked", False)),
    )

    span.set_attribute("usage", serialize_json(usage))


def set_tool_calls_attributes(span: Span, response: Optional[LLMResult]) -> None:
    """Set tool calls attributes on span from LLM response."""
    if not response or not response.generations or not response.generations[0]:
        return

    generation = response.generations[0][0]
    message = getattr(generation, "message", None)
    if not message:
        return

    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return

    content = getattr(message, "content", None)
    if content and isinstance(content, str) and content.strip():
        span.set_attribute("explanation", content)

    formatted_calls = []
    for tc in tool_calls:
        formatted_calls.append(
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
            }
        )
    if formatted_calls:
        span.set_attribute("toolCalls", serialize_json(formatted_calls))


def parse_tool_arguments(input_str: str) -> Optional[Dict[str, Any]]:
    """Parse tool arguments from JSON string or Python dict representation."""
    if not input_str:
        return None

    try:
        args = json.loads(input_str)
        return args if isinstance(args, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: try Python dict string representation (single quotes)
    try:
        args = ast.literal_eval(input_str)
        return args if isinstance(args, dict) else None
    except (ValueError, SyntaxError, TypeError) as e:
        logger.warning(
            "Failed to parse tool arguments from string: %s. Error: %s",
            input_str[:100],
            str(e),
        )
        return None


def _unwrap_tool_output(output: Any) -> Any:
    """Extract content from ToolMessage if wrapped by LangChain's BaseTool."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        return output

    if not isinstance(output, ToolMessage):
        return output

    content = output.content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass
    return content


def filter_output(output: Any) -> Any:
    """Return None if output is a NO_CONTENT trigger marker, otherwise return unchanged."""
    if output is None or is_no_content_marker(output):
        return None
    return output


def set_tool_result(span: Span, output: Any, attribute_name: str = "result") -> None:
    """Set tool result attribute on span."""
    actual = filter_output(_unwrap_tool_output(output))
    if actual is None:
        return
    if isinstance(actual, (dict, list)):
        span.set_attribute(attribute_name, serialize_json(actual))
    else:
        span.set_attribute(attribute_name, str(actual))


def get_tool_type_value(tool_type: Optional[str]) -> str:
    """Map tool_type metadata to toolType span attribute value."""
    mapping = {
        "process": "Process",
        "agent": "Agent",
        "api": "Api",
        "processorchestration": "agenticProcess",
        "escalation": "Escalation",
        "internal": "Internal",
        "ixp_extraction": "IxpExtraction",
        "mcp": "Mcp",
        "vs_escalation": "Escalation",
        "context_grounding": "ContextGrounding",
    }
    return mapping.get(tool_type or "", "Integration")


def build_task_url(task_id: int | str) -> str | None:
    """Build Action Center task URL from environment.

    UIPATH_URL already includes org/tenant in the path
    (e.g. https://alpha.uipath.com/org-id/tenant-id/).
    """
    from uipath.platform.common import UiPathConfig

    base_url = UiPathConfig.base_url
    if not base_url:
        return None
    url = f"{base_url.rstrip('/')}/actions_/tasks/{task_id}"
    return url


def set_process_job_info(span: Span, output: Any) -> None:
    """Set process job info attributes on span."""
    if not isinstance(output, dict):
        return
    job_id = output.get("job_id") or output.get("jobId") or output.get("JobId")
    if job_id:
        span.set_attribute("jobId", str(job_id))
    job_uri = (
        output.get("job_details_uri")
        or output.get("jobDetailsUri")
        or output.get("JobDetailsUri")
        or output.get("job_url")
        or output.get("jobUrl")
    )
    if job_uri:
        span.set_attribute("jobDetailsUri", str(job_uri))


def get_span_attachments(
    data: Optional[Dict[str, Any]],
    schema: Optional[Union[Dict[str, Any], Type[BaseModel]]],
    direction: Union[int, AttachmentDirection] = AttachmentDirection.NONE,
    provider: Union[int, AttachmentProvider] = AttachmentProvider.ORCHESTRATOR,
) -> Optional[List[SpanAttachment]]:
    """Extract span attachments from input data using JSON schema or Pydantic model.

    Converts input schema to a Pydantic model (if needed), extracts job attachments,
    and converts them to SpanAttachment objects for tracing.

    Args:
        input_data: The input data dictionary to extract attachments from
        input_schema: JSON schema definition (dict) or Pydantic model class
        direction: Attachment direction (int or AttachmentDirection enum)
        provider: Attachment provider (int or AttachmentProvider enum)

    Returns:
        List of SpanAttachment objects, or None if no attachments found
    """
    if not data or not schema:
        return None

    try:
        from uipath_langchain.agent.react.job_attachments import get_job_attachments
        from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
            create_model,
        )

        if isinstance(schema, dict):
            pydantic_model = create_model(schema)
        elif issubclass(schema, BaseModel):
            pydantic_model = schema
        else:
            return None

        job_attachments = get_job_attachments(pydantic_model, data)

        if not job_attachments:
            return None

        span_attachments = [
            SpanAttachment.model_validate(
                {
                    "id": str(att.id),
                    "file_name": att.full_name or "",
                    "mime_type": att.mime_type or "",
                    "provider": provider,
                    "direction": direction,
                }
            )
            for att in job_attachments
        ]

        return span_attachments if span_attachments else None

    except ImportError as e:
        logger.warning(
            "Failed to extract span attachments: uipath_langchain is not available. "
            "Install uipath_langchain to enable attachment extraction. Error: %s",
            str(e),
        )
        return None
    except Exception as e:
        logger.warning(
            "Failed to extract span attachments from input data. "
            "Tracing will continue without attachments. Error: %s",
            str(e),
            exc_info=True,
        )
        return None


def _coerce_to_dict(output: Any) -> Optional[Dict[str, Any]]:
    """Coerce output to a dict, deserializing from JSON string if needed.

    Args:
        output: The output data (dict, JSON string, or other)

    Returns:
        Dict if output is or contains a dict, None otherwise
    """
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _get_existing_attachments(
    existing_json: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    """Parse existing attachments from JSON string.

    Args:
        existing_json: JSON string containing existing attachments

    Returns:
        List of attachment dicts, or None if invalid/empty
    """
    if not existing_json:
        return None

    try:
        attachments = json.loads(existing_json)
        return attachments if isinstance(attachments, list) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _merge_attachments(
    existing: Optional[List[Dict[str, Any]]],
    new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge existing and new attachments.

    Args:
        existing: Existing attachment dicts (or None)
        new: New attachment dicts to add

    Returns:
        Merged list of attachments
    """
    if existing:
        return existing + new
    return new


def set_span_attachments(
    span: Union[Span, Dict[str, Any]],
    output: Any,
    output_schema: Optional[Union[Dict[str, Any], Type[BaseModel]]],
    direction: AttachmentDirection,
) -> None:
    """Set output attachments on span from tool output.

    Merges new attachments with any existing attachments on the span.

    Args:
        span: The span (Span object or dict) to set attachments on
        output: The tool output data
        output_schema: JSON schema or Pydantic model for the output
        direction: Attachment direction
    """
    if not output or not output_schema:
        return

    output_data = _coerce_to_dict(output)
    if not output_data:
        return

    new_attachments = get_span_attachments(
        output_data, output_schema, direction=direction
    )
    if not new_attachments:
        return

    new_attachments_list = [att.model_dump(by_alias=True) for att in new_attachments]

    if isinstance(span, dict):
        attributes = span.get("attributes", {})
        existing = _get_existing_attachments(attributes.get("attachments"))
        merged = _merge_attachments(existing, new_attachments_list)
        attributes["attachments"] = serialize_json(merged)
        span["attributes"] = attributes
    else:
        existing_json = (
            span.attributes.get("attachments") if hasattr(span, "attributes") else None
        )
        existing = _get_existing_attachments(existing_json)
        merged = _merge_attachments(existing, new_attachments_list)
        span.set_attribute("attachments", serialize_json(merged))


def set_context_grounding_results(span: Span, output: Any) -> None:
    """Set results, index_id, and output attachments on a context grounding span."""
    content = _unwrap_tool_output(output)

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

    if not content or not isinstance(content, dict):
        return

    raw = content.get("result") if isinstance(content, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    result_info: Dict[str, Any] = {}
    if "ID" in raw:
        result_info["id"] = raw["ID"]
    if "FullName" in raw:
        result_info["fileName"] = raw["FullName"]
    if "MimeType" in raw:
        result_info["mimeType"] = raw["MimeType"]
    if result_info:
        span.set_attribute("results", serialize_json(result_info))

    index_id = content.get("index_id") or content.get("indexId")
    if index_id:
        span.set_attribute("index_id", str(index_id))

    if not (result_info.get("id") and result_info.get("fileName")):
        return

    att = SpanAttachment.model_validate(
        {
            "id": str(result_info["id"]),
            "file_name": result_info.get("fileName", ""),
            "mime_type": result_info.get("mimeType", ""),
            "provider": AttachmentProvider.ORCHESTRATOR,
            "direction": AttachmentDirection.OUT,
        }
    )
    new_attachments = [att.model_dump(by_alias=True)]
    existing_json = (
        span.attributes.get("attachments") if hasattr(span, "attributes") else None
    )
    existing = _get_existing_attachments(existing_json)
    merged = _merge_attachments(existing, new_attachments)
    span.set_attribute("attachments", serialize_json(merged))
