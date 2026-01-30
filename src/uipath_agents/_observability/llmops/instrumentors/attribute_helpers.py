"""Attribute extraction and formatting helpers for span instrumentation."""

import json
import re
from typing import Any, Dict, Optional, Tuple

from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span

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

    if not token_usage:
        return

    usage = {
        "completionTokens": token_usage.get("completion_tokens", 0),
        "promptTokens": token_usage.get("prompt_tokens", 0),
        "totalTokens": token_usage.get("total_tokens", 0),
        "isByoExecution": False,
        "executionDeploymentType": None,
        "isPiiMasked": False,
        "llmCalls": 1,
    }
    span.set_attribute("usage", json.dumps(usage))


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
        span.set_attribute("toolCalls", json.dumps(formatted_calls))


def parse_tool_arguments(input_str: str) -> Optional[Dict[str, Any]]:
    """Parse tool arguments from JSON string."""
    if not input_str:
        return None
    try:
        args = json.loads(input_str)
        return args if isinstance(args, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def set_tool_result(span: Span, output: Any) -> None:
    """Set tool result attribute on span."""
    if output is None:
        return
    if isinstance(output, (dict, list)):
        span.set_attribute("result", json.dumps(output))
    else:
        span.set_attribute("result", str(output))


def get_tool_type_value(tool_type: Optional[str]) -> str:
    """Map tool_type to toolType attribute value."""
    if tool_type == "agent":
        return "Agent"
    elif tool_type == "process":
        return "Process"
    elif tool_type == "escalation":
        return "ActionCenter"
    return "Integration"


def set_escalation_task_info(span: Span, output: Any) -> None:
    """Set escalation task info attributes on span."""
    if not isinstance(output, dict):
        return
    task_id = output.get("task_id") or output.get("taskId") or output.get("id")
    if task_id:
        span.set_attribute("taskId", str(task_id))
    task_url = output.get("task_url") or output.get("taskUrl") or output.get("url")
    if task_url:
        span.set_attribute("taskUrl", str(task_url))


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
