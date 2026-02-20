"""Tool span instrumentor for LLMOps instrumentation."""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import UUID

from pydantic import BaseModel
from uipath.core.guardrails import GuardrailScope
from uipath.core.serialization import serialize_json
from uipath.eval.mocks.mockable import MOCKED_ANNOTATION_KEY
from uipath.tracing import AttachmentDirection, AttachmentProvider, SpanAttachment
from uipath_langchain.agent.guardrails.types import ExecutionStage

from ..span_hierarchy import SpanHierarchyManager
from ..spans import SpanKeys
from .attribute_helpers import (
    build_task_url,
    filter_output,
    get_tool_type_value,
    parse_tool_arguments,
    set_context_grounding_results,
    set_process_job_info,
    set_span_attachments,
    set_tool_result,
)
from .base import BaseSpanInstrumentor, InstrumentationState

logger = logging.getLogger(__name__)


class ToolSpanInstrumentor(BaseSpanInstrumentor):
    """Instruments tool events with spans: on_tool_start, on_tool_end, on_tool_error.

    Creates tool call spans with optional child spans for escalation, process,
    agent, or integration tools.

    Span hierarchy:
        AgentRun
        └── ToolCall (outer)
            └── [Escalation|Process|Agent|Integration] (child)
    """

    def __init__(
        self,
        state: InstrumentationState,
        close_container: Callable[[str, str], None],
    ) -> None:
        """Initialize Tool span instrumentor.

        Args:
            state: Shared instrumentation state
            close_container: Callback to close guardrail containers (scope, stage)
        """
        super().__init__(state)
        self._close_container = close_container

    def _interruptible_span_key(self, run_id: UUID) -> UUID:
        """Derive unique key for tool child span."""
        return SpanKeys.tool_child(run_id)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool start event."""
        try:
            tool_name = serialized.get("name", "unknown")
            tool_type = metadata.get("tool_type") if metadata else None
            tool_display_name = metadata.get("display_name") if metadata else None

            # MCP tools: construct full name as mcp-{slug}-tool-{name}
            if tool_type == "mcp" and metadata:
                slug = metadata.get("slug")
                if slug:
                    sanitized_slug = slug.replace("-", "_")
                    tool_name = f"mcp-{sanitized_slug}-tool-{tool_name}"

            output_schema = metadata.get("output_schema") if metadata else None
            if output_schema is not None:
                self._state.tool_output_schemas[run_id] = output_schema

            # Resume mode: skip duplicate span for re-invoked tool
            if (
                self._state.resume_tool_name
                and tool_name == self._state.resume_tool_name
            ):
                logger.debug("Resume mode: skipping span creation for %s", tool_name)
                self._state.resume_tool_name = None
                self._state.reinvoked_tool_run_ids.add(run_id)
                return

            # Get call_id from kwargs, fall back to metadata (set by escalation_wrapper)
            call_id = kwargs.get("tool_call_id")
            if not call_id and metadata:
                call_id = metadata.get("_call_id")

            # Get arguments from input_str, fall back to metadata
            arguments = parse_tool_arguments(input_str)
            if not arguments and metadata:
                arguments = metadata.get("_call_args")

            tool_type_value = get_tool_type_value(tool_type)
            args_schema = metadata.get("args_schema") if metadata else None

            # Check if tool span was created early by tool_pre guardrails
            # Only reuse if the flag indicates it was created by guardrails
            if self._state.current_tool_span and self._state.tool_span_from_guardrail:
                span = self._state.current_tool_span
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("toolType", tool_type_value)
                if call_id:
                    span.set_attribute("call_id", call_id)
                if arguments:
                    span.set_attribute("input", json.dumps(arguments))
                self._spans[run_id] = span
                # Clear the flag after reuse - span was consumed by on_tool_start
                self._state.tool_span_from_guardrail = False
                self._close_container(GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION)
                self._span_factory.upsert_span_started(span)
            else:
                parent = self._state.current_llm_span or self._state.get_span_or_root(
                    parent_run_id
                )
                span = self._span_factory.start_tool_call(
                    tool_name,
                    tool_type_value=tool_type_value,
                    arguments=arguments,
                    call_id=call_id,
                    parent_span=parent,
                    args_schema=args_schema,
                )
                span.set_attribute("tool.name", tool_name)
                if call_id:
                    span.set_attribute("call_id", call_id)
                if arguments:
                    span.set_attribute("input", json.dumps(arguments))
                self._spans[run_id] = span
                # Set current_tool_span for HITL to access if needed
                self._state.current_tool_span = span

            SpanHierarchyManager.push(run_id, span)

            # Create child span for typed tools
            if tool_type:
                child_span = None
                if tool_type == "escalation" and tool_display_name:
                    channel_type = metadata.get("channel_type") if metadata else None
                    child_span = self._span_factory.start_escalation_tool(
                        app_name=tool_display_name,
                        arguments=arguments,
                        channel_type=channel_type,
                        parent_span=span,
                        args_schema=args_schema,
                    )
                    self._state.escalation_run_ids.add(run_id)
                elif tool_type == "agent" and tool_display_name:
                    child_span = self._span_factory.start_agent_tool(
                        agent_name=tool_display_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                    self._state.agent_run_ids.add(run_id)
                elif (
                    tool_type in ("process", "api", "processorchestration")
                    and tool_display_name
                ):
                    child_span = self._span_factory.start_process_tool(
                        process_name=tool_display_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                    self._state.process_run_ids.add(run_id)
                elif tool_type == "integration":
                    child_span = self._span_factory.start_integration_tool(
                        tool_name=tool_display_name or tool_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                elif tool_type == "ixp_extraction" and tool_display_name:
                    project_name = metadata.get("project_name") if metadata else None
                    version_tag = metadata.get("version_tag") if metadata else None
                    child_span = self._span_factory.start_ixp_tool(
                        tool_name=tool_display_name,
                        arguments=arguments,
                        project_name=project_name,
                        version_tag=version_tag,
                        parent_span=span,
                    )
                    self._state.ixp_extraction_run_ids.add(run_id)
                elif tool_type == "vs_escalation":
                    ixp_tool_id = metadata.get("ixp_tool_id") if metadata else None
                    vs_storage_bucket_name = (
                        metadata.get("storage_bucket_name") if metadata else None
                    )
                    child_span = self._span_factory.start_vs_escalation_tool(
                        tool_name=tool_display_name or tool_name,
                        arguments=arguments,
                        ixp_tool_id=ixp_tool_id,
                        storage_bucket_name=vs_storage_bucket_name,
                        parent_span=span,
                    )
                    self._state.vs_escalation_run_ids.add(run_id)
                elif tool_type == "context_grounding":
                    assert metadata is not None  # tool_type came from metadata
                    query = (
                        (arguments or {}).get("query")
                        or metadata.get("static_query")
                        or ""
                    )
                    input_attachments = None
                    file_extension = None
                    att_raw = (arguments or {}).get("attachment")
                    if isinstance(att_raw, dict) and att_raw.get("ID"):
                        mime_type = att_raw.get("MimeType")

                        input_attachments = [
                            SpanAttachment.model_validate(
                                {
                                    "id": str(att_raw["ID"]),
                                    "file_name": att_raw.get("FullName", ""),
                                    "mime_type": mime_type,
                                    "provider": AttachmentProvider.ORCHESTRATOR,
                                    "direction": AttachmentDirection.IN,
                                }
                            )
                        ]

                        file_extension = mime_type

                    child_span = self._span_factory.start_context_grounding_tool(
                        tool_name=tool_display_name or tool_name,
                        retrieval_mode=metadata.get("retrieval_mode", "SemanticSearch"),
                        query=query,
                        output_columns=metadata.get("output_columns"),
                        web_search_grounding=metadata.get("web_search_grounding"),
                        citation_mode=metadata.get("citation_mode"),
                        number_of_results=metadata.get("number_of_results"),
                        file_extension=file_extension,
                        parent_span=span,
                        input_attachments=input_attachments,
                    )
                    self._state.context_grounding_run_ids.add(run_id)
                elif tool_type == "internal" and tool_display_name:
                    child_span = self._span_factory.start_internal_tool(
                        tool_name=tool_display_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                elif tool_type == "mcp":
                    child_span = self._span_factory.start_mcp_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                    self._state.mcp_run_ids.add(run_id)

                if child_span:
                    self._spans[self._interruptible_span_key(run_id)] = child_span
                    SpanHierarchyManager.push(run_id, child_span)
                    if metadata and "_span_context" in metadata:
                        metadata["_span_context"]["parent_span_id"] = (
                            child_span.get_span_context().span_id
                        )
                    if tool_type in (
                        "escalation",
                        "process",
                        "agent",
                        "internal",
                        "api",
                        "processorchestration",
                        "ixp_extraction",
                        "vs_escalation",
                        "context_grounding",
                    ):
                        self._state.pending_tool_name = tool_name
                        self._state.pending_tool_span = span
                        self._state.pending_process_span = child_span

        except Exception:
            logger.exception("Error in on_tool_start callback")

    def _set_ixp_extraction_result_attrs(self, span: Any, output: Any) -> None:
        """Set extractionId and documentId on IXP extraction span from completion result."""
        if not isinstance(output, dict):
            return
        extraction_result = output.get("extractionResult") or output.get(
            "extraction_result"
        )
        if isinstance(extraction_result, dict):
            doc_id = extraction_result.get("DocumentId") or extraction_result.get(
                "document_id"
            )
            if doc_id:
                span.set_attribute("documentId", str(doc_id))
        operation_id = output.get("operationId") or output.get("operation_id")
        if operation_id:
            span.set_attribute("extractionId", str(operation_id))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool end event."""
        try:
            output_schema = self._state.tool_output_schemas.pop(run_id, None)
            output = output.content if hasattr(output, "content") else output

            # Handle resumed tool completion
            if run_id in self._state.reinvoked_tool_run_ids:
                self._state.reinvoked_tool_run_ids.discard(run_id)
                self._upsert_resumed_spans_on_completion(output, output_schema)
                return

            is_cg = run_id in self._state.context_grounding_run_ids

            # Close child span first (inner), then tool span (outer)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                if hasattr(child_span, "attributes"):
                    if child_span.attributes.get(MOCKED_ANNOTATION_KEY) and hasattr(
                        child_span, "name"
                    ):
                        child_span.update_name(f"Simulated result: {child_span.name}")
                SpanHierarchyManager.pop(run_id)
                set_tool_result(child_span, output, "output")
                if run_id in self._state.mcp_run_ids:
                    set_tool_result(child_span, output)
                    self._state.mcp_run_ids.discard(run_id)
                if run_id in self._state.process_run_ids:
                    set_process_job_info(child_span, output)
                    self._state.process_run_ids.discard(run_id)
                if run_id in self._state.agent_run_ids:
                    set_process_job_info(child_span, output)
                    self._state.agent_run_ids.discard(run_id)
                if run_id in self._state.ixp_extraction_run_ids:
                    self._set_ixp_extraction_result_attrs(child_span, output)
                    self._state.ixp_extraction_run_ids.discard(run_id)
                if run_id in self._state.vs_escalation_run_ids:
                    self._state.vs_escalation_run_ids.discard(run_id)
                if run_id in self._state.context_grounding_run_ids:
                    set_context_grounding_results(child_span, output)
                    output_schema = self._state.tool_output_schemas.get(run_id)
                    if isinstance(output, str) and output_schema:
                        try:
                            parsed_output = json.loads(output)
                            set_span_attachments(
                                child_span,
                                parsed_output,
                                output_schema,
                                AttachmentDirection.OUT,
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass
                    self._state.context_grounding_run_ids.discard(run_id)
                self._span_factory.end_span_ok(child_span)
                if child_span == self._state.pending_process_span:
                    self._state.pending_process_span = None

            span = self._spans.pop(run_id, None)
            if span:
                SpanHierarchyManager.pop(run_id)
                if is_cg:
                    raw = output.get("result") if isinstance(output, dict) else None
                    if isinstance(raw, dict):
                        result_info = {}
                        if "ID" in raw:
                            result_info["id"] = raw["ID"]
                        if "FullName" in raw:
                            result_info["fileName"] = raw["FullName"]
                        if "MimeType" in raw:
                            result_info["mimeType"] = raw["MimeType"]
                        if result_info:
                            span.set_attribute("result", serialize_json(result_info))
                else:
                    set_tool_result(
                        span, output, "output"
                    )  # ugly fix for stupid problem...
                set_span_attachments(
                    span, output, output_schema, AttachmentDirection.OUT
                )
                self._span_factory.end_span_ok(span)
                if span == self._state.pending_tool_span:
                    self._state.pending_tool_span = None
                    self._state.pending_tool_name = None

        except Exception:
            logger.exception("Error in on_tool_end callback")

    def _upsert_resumed_spans_on_completion(
        self,
        output: Any,
        output_schema: Optional[Union[Dict[str, Any], Type[BaseModel]]],
    ) -> None:
        """Upsert resumed tool/process spans when tool completes."""
        if not self._state.resumed_trace_id:
            return

        output = filter_output(output)

        # Upsert process span first (inner)
        is_escalation = False
        if self._state.resumed_process_span_data:
            span_attrs = self._state.resumed_process_span_data.setdefault(
                "attributes", {}
            )
            span_type = span_attrs.get("span_type", "") or span_attrs.get("type", "")
            is_escalation = span_type == "escalationTool"

            if output is not None:
                result = (
                    json.dumps(output)
                    if isinstance(output, (dict, list))
                    else str(output)
                )

                if is_escalation:
                    filtered_output = {
                        k: output[k] for k in ("outcome", "output") if k in output
                    }
                    result = json.dumps(filtered_output)

                span_attrs["result"] = result

            self._span_factory.upsert_span_complete_by_data(
                trace_id=self._state.resumed_trace_id,
                span_data=self._state.resumed_process_span_data,
            )
            logger.debug(
                "Upserted resumed process span %s on tool completion",
                self._state.resumed_process_span_data.get("name", "unknown"),
            )

        # Upsert tool span (outer)
        if self._state.resumed_tool_span_data:
            tool_attrs = self._state.resumed_tool_span_data.setdefault("attributes", {})

            # Ensure attributes use names exporter expects (call_id, input)
            if tool_attrs.get("callId") and not tool_attrs.get("call_id"):
                tool_attrs["call_id"] = tool_attrs["callId"]
            if tool_attrs.get("arguments") and not tool_attrs.get("input"):
                tool_attrs["input"] = tool_attrs["arguments"]

            if output is not None:
                result = (
                    json.dumps(output)
                    if isinstance(output, (dict, list))
                    else str(output)
                )

                if is_escalation:
                    filtered_output = {
                        k: output[k] for k in ("outcome", "output") if k in output
                    }
                    result = json.dumps(filtered_output)

                tool_attrs["output"] = result
                set_span_attachments(
                    self._state.resumed_tool_span_data,
                    output,
                    output_schema,
                    AttachmentDirection.OUT,
                )

            self._span_factory.upsert_span_complete_by_data(
                trace_id=self._state.resumed_trace_id,
                span_data=self._state.resumed_tool_span_data,
            )
            logger.debug(
                "Upserted resumed tool span %s on tool completion",
                self._state.resumed_tool_span_data.get("name", "unknown"),
            )

        # End original OTEL spans (preserved across suspend/resume) so file
        # exporter sees the result. These were saved during cleanup() before
        # the first execution returned SUSPENDED.
        is_cg = False
        if self._state.resumed_process_span_data:
            cg_type = self._state.resumed_process_span_data.get("attributes", {}).get(
                "span_type", ""
            ) or self._state.resumed_process_span_data.get("attributes", {}).get(
                "type", ""
            )
            is_cg = cg_type == "contextGroundingTool"

        suspended_process = self._state.suspended_process_span
        if suspended_process:
            set_tool_result(suspended_process, output, "output")
            if is_cg:
                set_context_grounding_results(suspended_process, output)
                if output_schema:
                    set_span_attachments(
                        suspended_process,
                        output,
                        output_schema,
                        AttachmentDirection.OUT,
                    )
            self._span_factory.end_span_ok(suspended_process)
            self._state.suspended_process_span = None

        suspended_tool = self._state.suspended_tool_span
        if suspended_tool:
            if is_cg:
                raw = output.get("result") if isinstance(output, dict) else None
                if isinstance(raw, dict):
                    result_info = {}
                    if "ID" in raw:
                        result_info["id"] = raw["ID"]
                    if "FullName" in raw:
                        result_info["fileName"] = raw["FullName"]
                    if "MimeType" in raw:
                        result_info["mimeType"] = raw["MimeType"]
                    if result_info:
                        suspended_tool.set_attribute(
                            "result", serialize_json(result_info)
                        )
            else:
                set_tool_result(suspended_tool, output, "output")
            set_span_attachments(
                suspended_tool, output, output_schema, AttachmentDirection.OUT
            )
            self._span_factory.end_span_ok(suspended_tool)
            self._state.suspended_tool_span = None

        self._state.resumed_process_span_data = None
        self._state.resumed_tool_span_data = None

    def _is_graph_interrupt(self, error: BaseException) -> bool:
        """Check if the error is a GraphInterrupt (suspend signal)."""
        error_str = str(error)
        error_type = type(error).__name__
        return error_type == "GraphInterrupt" or error_str.startswith("GraphInterrupt(")

    def _handle_graph_interrupt(self, run_id: UUID, error: BaseException) -> None:
        """Extract task metadata from interrupt payload and upsert spans."""
        child_key = self._interruptible_span_key(run_id)
        child_span = self._spans.get(child_key)

        # Set type-specific attributes from the interrupt payload
        if child_span and run_id in self._state.escalation_run_ids:
            self._set_escalation_interrupt_attrs(child_span, error)
        elif child_span and run_id in self._state.vs_escalation_run_ids:
            self._set_vs_escalation_interrupt_attrs(child_span, error)
        elif child_span and (
            run_id in self._state.process_run_ids or run_id in self._state.agent_run_ids
        ):
            self._set_process_interrupt_attrs(child_span, error)

        # Upsert both spans as suspended
        if child_span:
            self._span_factory.upsert_span_suspended(child_span)
        tool_span = self._spans.get(run_id)
        if tool_span:
            self._span_factory.upsert_span_suspended(tool_span)

    def _set_escalation_interrupt_attrs(self, span: Any, error: BaseException) -> None:
        """Extract task metadata from escalation interrupt and set span attributes."""
        interrupts = error.args[0] if error.args else None
        if not interrupts:
            return

        value = getattr(interrupts[0], "value", None)
        if not value:
            return

        action = getattr(value, "action", None)
        if action:
            task_id = getattr(action, "id", None)
            if task_id:
                span.set_attribute("taskId", str(task_id))
                task_url = build_task_url(task_id)
                if task_url:
                    span.set_attribute("taskUrl", task_url)

        recipient = getattr(value, "recipient", None)
        if recipient and getattr(recipient, "display_name", None):
            span.set_attribute("assignedTo", recipient.display_name)

    def _set_vs_escalation_interrupt_attrs(
        self, span: Any, error: BaseException
    ) -> None:
        """Extract operation_id and taskUrl from VS escalation interrupt and set span attributes."""
        interrupts = error.args[0] if error.args else None
        if not interrupts:
            return

        value = getattr(interrupts[0], "value", None)
        if not value:
            return

        extraction_validation = getattr(value, "extraction_validation", None)
        if extraction_validation:
            operation_id = getattr(extraction_validation, "operation_id", None)
            if operation_id:
                span.set_attribute("operationId", str(operation_id))

        task_url = getattr(value, "task_url", None)
        if task_url:
            span.set_attribute("taskUrl", str(task_url))

    def _set_process_interrupt_attrs(self, span: Any, error: BaseException) -> None:
        """Extract job metadata from process interrupt and set span attributes."""
        interrupts = error.args[0] if error.args else None
        if not interrupts:
            return

        value = getattr(interrupts[0], "value", None)
        if not value:
            return

        job = getattr(value, "job", None)
        if not job:
            return

        job_key = getattr(job, "key", None)
        if job_key:
            span.set_attribute("jobId", str(job_key))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool error event."""
        try:
            self._state.tool_output_schemas.pop(run_id, None)

            if run_id in self._state.reinvoked_tool_run_ids:
                self._state.reinvoked_tool_run_ids.discard(run_id)
                return

            # GraphInterrupt = suspend signal; extract task metadata and upsert spans
            if self._is_graph_interrupt(error):
                self._handle_graph_interrupt(run_id, error)
                return

            exc = error if isinstance(error, Exception) else Exception(str(error))

            # Close child span first (inner), then tool span (outer)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                SpanHierarchyManager.pop(run_id)
                self._span_factory.end_span_error(child_span, exc)

            span = self._spans.pop(run_id, None)
            if span:
                SpanHierarchyManager.pop(run_id)
                self._span_factory.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_tool_error callback")
