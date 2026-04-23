"""Trace span emission for memory recall, matching the Temporal backend's
DynamicFewShotWorkflow (DynamicFewShotWorkflow.cs:29-52).

Emits two spans:
  - "Find previous memories" (agentMemoryLookup) — parent span
  - "Apply dynamic few shot" (applyDynamicFewShot) — child span
"""

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _generate_span_id() -> str:
    """Generate a 16-char hex span ID."""
    return secrets.token_hex(8)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _build_span(
    name: str,
    span_type: str,
    span_id: str,
    trace_id: str,
    parent_id: str | None,
    start_time: str,
    end_time: str,
    attributes: dict[str, Any],
    status: int = 1,  # 1=OK
) -> dict[str, Any]:
    """Build a span payload matching LlmOpsSpan structure."""
    return {
        "Id": span_id,
        "TraceId": trace_id,
        "ParentId": parent_id,
        "Name": name,
        "StartTime": start_time,
        "EndTime": end_time,
        "Attributes": json.dumps(attributes),
        "Status": status,
        "SpanType": span_type,
        "Source": 4,  # Python SDK source
        "OrganizationId": os.environ.get("UIPATH_ORGANIZATION_ID"),
        "TenantId": os.environ.get("UIPATH_TENANT_ID"),
        "FolderKey": os.environ.get("UIPATH_FOLDER_KEY"),
        "ProcessKey": os.environ.get("UIPATH_PROCESS_UUID"),
        "JobKey": os.environ.get("UIPATH_JOB_KEY"),
    }


async def emit_memory_recall_spans(
    sdk: Any,
    trace_id: str,
    parent_span_id: str | None,
    memory_space_id: str,
    memory_space_name: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None,
    results_count: int,
    injection: str,
    start_time: str,
    end_time: str,
    error: str | None = None,
) -> None:
    """Emit 'Find previous memories' and 'Apply dynamic few shot' trace spans.

    Mirrors DynamicFewShotWorkflow.cs:29-52 in the Temporal backend.
    """
    lookup_span_id = _generate_span_id()
    fewshot_span_id = _generate_span_id()
    status = 1 if not error else 2  # 1=OK, 2=ERROR

    # Span 1: "Find previous memories" (parent)
    # Ref: SpanName.MemorySpaceLookup, SpanType.AgentMemoryLookup
    lookup_attrs: dict[str, Any] = {
        "memorySpaceId": memory_space_id,
        "memorySpaceName": memory_space_name,
        "strategy": "DynamicFewShotPrompt",
        "memoryItemsMatched": results_count,
    }
    if injection:
        lookup_attrs["result"] = injection
    if error:
        lookup_attrs["error"] = error

    lookup_span = _build_span(
        name="Find previous memories",
        span_type="agentMemoryLookup",
        span_id=lookup_span_id,
        trace_id=trace_id,
        parent_id=parent_span_id,
        start_time=start_time,
        end_time=end_time,
        attributes=lookup_attrs,
        status=status,
    )

    # Span 2: "Apply dynamic few shot" (child of lookup)
    # Ref: SpanName.DynamicFewShotPrompt, SpanType.ApplyDynamicFewShot
    fewshot_attrs: dict[str, Any] = {
        "memorySpaceId": memory_space_id,
        "memorySpaceName": memory_space_name,
    }
    if request_payload:
        fewshot_attrs["request"] = request_payload
    if response_payload:
        fewshot_attrs["response"] = response_payload

    fewshot_span = _build_span(
        name="Apply dynamic few shot",
        span_type="applyDynamicFewShot",
        span_id=fewshot_span_id,
        trace_id=trace_id,
        parent_id=lookup_span_id,
        start_time=start_time,
        end_time=end_time,
        attributes=fewshot_attrs,
        status=status,
    )

    # Send both spans to LLMOps
    logger.warning(
        "Emitting memory trace spans: trace_id=%s, parent=%s, lookup_id=%s",
        trace_id,
        parent_span_id,
        lookup_span_id,
    )
    try:
        base_url = os.environ.get("UIPATH_TRACE_BASE_URL") or (
            os.environ.get("UIPATH_URL", "").rstrip("/") + "/llmopstenant_"
        )
        url = f"{base_url}/api/Traces/spans?traceId={trace_id}&source=Robots"

        token = os.environ.get("UIPATH_ACCESS_TOKEN", "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        import httpx
        from uipath._utils._ssl_context import get_httpx_client_kwargs

        async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
            for span in [lookup_span, fewshot_span]:
                resp = await client.post(url, json=[span], headers=headers)
                if resp.status_code != 200:
                    logger.warning(
                        "Failed to emit span '%s': %s %s",
                        span["Name"],
                        resp.status_code,
                        resp.text[:200],
                    )
    except Exception as e:
        logger.warning("Failed to emit memory recall trace spans: %s", e)
