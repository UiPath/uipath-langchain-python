"""Telemetry wrapper for UiPath runtimes.

Provides telemetry/tracing capabilities via composition (wrapper pattern)
rather than inheritance (mixin pattern).

Manual instrumentation is always enabled for dual instrumentation:
- Manual spans (agentRun, llmCall, toolCall) → LLMOps (user-facing)
- OpenInference spans → AppInsights (debugging)

Supports interruptible process trace context preservation:
- On SUSPENDED: upserts span with RUNNING status, saves trace context
- On resume: restores trace context, continues same trace
"""

import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, AsyncGenerator, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanContext, TraceFlags
from uipath._cli._utils._common import get_claim_from_token
from uipath.agent.models.agent import AgentDefinition
from uipath.platform.common import UiPathConfig
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeProtocol,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorContract
from uipath.runtime.events import UiPathRuntimeEvent
from uipath.runtime.schema import UiPathRuntimeSchema
from uipath.tracing import SpanStatus

from uipath_agents._errors import ExceptionMapper

from ..agent_graph_builder.config import get_execution_type
from .event_emitter import (
    AgentRunEvent,
    TelemetryEventEmitter,
)
from .llmops import (
    LlmOpsInstrumentationCallback,
    LlmOpsSpanFactory,
    PendingSpanData,
    TraceContextData,
    TraceContextStorage,
    reference_id_context,
)
from .llmops.spans.spans_schema.base import uipath_source_context

logger = logging.getLogger(__name__)


class InstrumentedRuntime:
    """Wrapper that adds instrumentation to any UiPathRuntimeProtocol implementation.

    Uses composition pattern to wrap any runtime with tracing capabilities.
    Manages agent span lifecycle and updates the callback before each execution.

    The callback is passed to the delegate runtime via constructor (not context vars),
    ensuring it persists across debug/chat re-executions where the same runtime
    instance is executed multiple times.

    Supports trace context preservation across suspend/resume cycles:
    - On SUSPENDED: upserts span with RUNNING status and persists trace context
    - On resume: loads trace context and continues the same trace

    Example:
        span_factory = LlmOpsSpanFactory()
        callback = LlmOpsInstrumentationCallback(span_factory)
        storage = SqliteTraceContextStorage(sqlite_storage)
        base_runtime = AgentsLangGraphRuntime(graph, callbacks=[callback])
        instrumented = InstrumentedRuntime(
            base_runtime, span_factory, callback, trace_context_storage=storage
        )
        result = await instrumented.execute(input_data)
    """

    def __init__(
        self,
        delegate: UiPathRuntimeProtocol,
        span_factory: LlmOpsSpanFactory,
        callback: LlmOpsInstrumentationCallback,
        runtime_context: UiPathRuntimeContext,
        event_emitter: TelemetryEventEmitter | None = None,
        agent_definition: AgentDefinition | None = None,
        trace_context_storage: TraceContextStorage | None = None,
    ):
        """Initialize the instrumented runtime.

        Args:
            delegate: The runtime to wrap with instrumentation
            span_factory: LlmOpsSpanFactory for creating spans
            callback: Callback for LangChain event instrumentation
            runtime_context: Runtime context containing command and environment info
            event_emitter: Optional emitter for Application Insights telemetry events
            agent_definition: Optional agent metadata for span attributes and telemetry enrichment
            trace_context_storage: Optional storage for trace context preservation
                across suspend/resume cycles. If None, trace context is not preserved.
        """
        self._delegate = delegate
        self._span_factory = span_factory
        self._callback = callback
        self._event_emitter = event_emitter
        self._agent_definition = agent_definition
        self._trace_context_storage = trace_context_storage
        self._agent_run_id = str(uuid.uuid4())
        self._runtime_context = runtime_context

    @property
    def delegate(self) -> UiPathRuntimeProtocol:
        return self._delegate

    async def execute(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent with telemetry instrumentation.

        Handles trace context preservation for interruptible agents:
        - Checks for saved trace context (resume scenario)
        - On SUSPENDED: upserts span with RUNNING status, saves context
        - On completion: clears saved trace context

        Args:
            input: Input data for the agent
            options: Execution options

        Returns:
            Execution result with status
        """
        start_time = time.time()
        # Check for existing trace context (resume scenario)
        saved_context = await self._load_trace_context_if_resuming()

        async with self._agent_span_context(
            input, saved_context, start_time
        ) as agent_span:
            try:
                result = await self._delegate.execute(input, options)
            except Exception as e:
                if saved_context:
                    self._handle_resume_error(saved_context, e)
                raise

            duration_ms = int((time.time() - start_time) * 1000)

            if result.status == UiPathRuntimeStatus.SUSPENDED:
                await self._handle_suspended(agent_span)
            elif result.status == UiPathRuntimeStatus.SUCCESSFUL:
                if saved_context:
                    self._handle_resume_complete(saved_context)
                self._emit_output_if_successful(result, duration_ms, agent_span)
                await self._clear_trace_context()
            elif result.status == UiPathRuntimeStatus.FAULTED:
                if saved_context and result.error:
                    self._handle_resume_error(saved_context, result.error)
                await self._clear_trace_context()
            else:
                raise ValueError(f"Unexpected runtime status: {result.status}")

            return result

    async def stream(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution with telemetry instrumentation.

        Handles trace context preservation for interruptible agents.

        Args:
            input: Input data for the agent
            options: Stream options

        Yields:
            Runtime events during execution
        """
        start_time = time.time()
        saved_context = await self._load_trace_context_if_resuming()

        async with self._agent_span_context(
            input, saved_context, start_time
        ) as agent_span:
            final_result: Optional[UiPathRuntimeResult] = None

            try:
                async for event in self._delegate.stream(input, options):
                    if isinstance(event, UiPathRuntimeResult):
                        final_result = event
                    yield event
            except Exception as e:
                if saved_context:
                    self._handle_resume_error(saved_context, e)
                raise

            if final_result:
                duration_ms = int((time.time() - start_time) * 1000)

                if final_result.status == UiPathRuntimeStatus.SUSPENDED:
                    await self._handle_suspended(agent_span)
                elif final_result.status == UiPathRuntimeStatus.SUCCESSFUL:
                    if saved_context:
                        self._handle_resume_complete(saved_context)
                    self._emit_output_if_successful(
                        final_result, duration_ms, agent_span
                    )
                    await self._clear_trace_context()
                elif final_result.status == UiPathRuntimeStatus.FAULTED:
                    if saved_context and final_result.error:
                        self._handle_resume_error(saved_context, final_result.error)
                    await self._clear_trace_context()
                else:
                    raise ValueError(
                        f"Unexpected runtime status: {final_result.status}"
                    )

    async def get_schema(self) -> UiPathRuntimeSchema:
        return await self._delegate.get_schema()

    async def dispose(self) -> None:
        await self._delegate.dispose()

    def get_agent_model(self) -> str | None:
        """Get the agent's configured LLM model from the delegate runtime.

        Delegates to the wrapped runtime's get_agent_model() if it exists.
        This allows the eval runtime to traverse through wrapper chains.

        Returns:
            The model name (e.g., 'gpt-4o-2024-11-20'), or None if not found.
        """
        if hasattr(self._delegate, "get_agent_model"):
            return self._delegate.get_agent_model()
        return None

    @asynccontextmanager
    async def _agent_span_context(
        self,
        input_data: Any = None,
        saved_context: Optional[TraceContextData] = None,
        start_time: Optional[float] = None,
    ) -> AsyncGenerator[Span, None]:
        """Context manager for agent span lifecycle.

        Creates or restores agent span and updates the callback to use it as root.
        The callback was already passed to delegate via constructor,
        so it will automatically receive events during execution.

        Args:
            input_data: Input data for the agent (used for new spans)
            saved_context: Previously saved trace context (resume scenario)
            start_time: Start time for telemetry tracking

        Yields:
            The agent span for the execution
        """
        if saved_context:
            # Resume: restore original trace context
            async with self._restore_trace_context(saved_context) as agent_span:
                self._callback.set_agent_span(
                    agent_span, uuid.UUID(self._agent_run_id), prompts_captured=True
                )

                # Rebuild enriched properties so post-resume events
                # (e.g. guardrail escalation approved/rejected) still carry
                # agent and execution metadata.
                if self._event_emitter:
                    self._build_and_set_enriched_properties(agent_span)

                pending_tool = saved_context.get("pending_tool_name")
                pending_tool_span = saved_context.get("pending_tool_span")
                if pending_tool:
                    pending_process_span = saved_context.get("pending_process_span")
                    self._callback.set_resume_context(
                        tool_name=pending_tool,
                        trace_id=saved_context.get("trace_id"),
                        tool_span_data=dict(pending_tool_span)
                        if pending_tool_span
                        else None,
                        process_span_data=dict(pending_process_span)
                        if pending_process_span
                        else None,
                    )
                pending_escalation = saved_context.get("pending_escalation_span")
                pending_guardrail_hitl_evaluation_span = saved_context.get(
                    "pending_guardrail_hitl_evaluation_span"
                )
                pending_guardrail_hitl_container_span = saved_context.get(
                    "pending_guardrail_hitl_container_span"
                )
                pending_llm_span = saved_context.get("pending_llm_span")
                if (
                    pending_escalation
                    and pending_guardrail_hitl_evaluation_span
                    and pending_guardrail_hitl_container_span
                ):
                    self._callback.set_escalation_resume_context(
                        trace_id=saved_context.get("trace_id", ""),
                        escalation_span_data=dict(pending_escalation),
                        hitl_guardrail_span_data=dict(
                            pending_guardrail_hitl_evaluation_span
                        ),
                        hitl_guardrail_container_span_data=dict(
                            pending_guardrail_hitl_container_span
                        ),
                        llm_span_data=dict(pending_llm_span)
                        if pending_llm_span
                        else None,
                        tool_span_data=dict(pending_tool_span)
                        if pending_tool_span
                        else None,
                    )
                try:
                    yield agent_span
                finally:
                    self._callback.cleanup()
                    if self._event_emitter:
                        self._event_emitter.cleanup()
        else:
            # Normal: create new trace context
            agent_name = self._get_agent_name()
            agent_id = self._get_agent_id()
            system_prompt, user_prompt = self._get_prompts()
            input_schema, output_schema = self._get_schemas()

            is_conversational = (
                self._agent_definition.is_conversational
                if self._agent_definition
                else False
            )

            with self._span_factory.start_agent_run(
                agent_name=agent_name,
                agent_id=agent_id,
                is_conversational=is_conversational or False,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                input_data=self._normalize_input(input_data),
                input_schema=input_schema,
                output_schema=output_schema,
                source=get_execution_type(self._runtime_context).value,
            ) as agent_span:
                prompts_captured = system_prompt is not None or user_prompt is not None
                self._callback.set_agent_span(
                    agent_span, uuid.UUID(self._agent_run_id), prompts_captured
                )

                if self._event_emitter:
                    self._event_emitter.set_agent_info(agent_name, agent_id)

                    enriched_properties = self._build_and_set_enriched_properties(
                        agent_span
                    )
                    self._event_emitter.track_event(
                        AgentRunEvent.STARTED, enriched_properties
                    )

                try:
                    yield agent_span
                except Exception as e:
                    if self._event_emitter:
                        duration_ms = (
                            int((time.time() - start_time) * 1000)
                            if start_time
                            else None
                        )

                        mapped_exc = ExceptionMapper.map_runtime(e)
                        error_info = mapped_exc.error_info

                        # Capture full traceback for telemetry debugging
                        # Most exceptions have include_traceback=False, so we manually capture it here
                        error_traceback = traceback.format_exc()

                        base_properties: Dict[str, Any] = {
                            "AgentName": agent_name,
                            "Status": "Failed",
                            "Timestamp": datetime.now(timezone.utc).isoformat(),
                            "ErrorMessage": str(e)[:500],
                            "ErrorType": type(e).__name__,
                            "ErrorCode": error_info.code,
                            "ErrorTitle": error_info.title,
                            "ErrorCategory": error_info.category.value,
                            "ErrorTraceback": error_traceback,
                        }

                        if agent_id:
                            base_properties["AgentId"] = agent_id
                        if duration_ms is not None:
                            base_properties["DurationMs"] = duration_ms

                        enriched_properties = self._get_enriched_properties(
                            base_properties
                        )

                        self._event_emitter.track_event(
                            AgentRunEvent.FAILED, enriched_properties
                        )
                    raise
                finally:
                    self._callback.cleanup()
                    if self._event_emitter:
                        self._event_emitter.cleanup()

    @asynccontextmanager
    async def _restore_trace_context(
        self, saved_context: TraceContextData
    ) -> AsyncGenerator[Span, None]:
        """Restore a previously saved trace context for resume.

        Restores the original agent span context so new spans created
        during resume become children of the original agent span (not
        nested under a new child span).

        Also restores the reference_id ContextVar so child spans inherit it.

        Args:
            saved_context: Previously saved trace context data

        Yields:
            A NonRecordingSpan representing the restored context
        """
        trace_id = int(saved_context["trace_id"], 16)
        span_id = int(saved_context["span_id"], 16)

        restored_span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(0x01),  # Sampled
        )

        # Create NonRecordingSpan to represent the restored agent span
        # This doesn't create a new span - it restores the original context
        restored_span = trace.NonRecordingSpan(restored_span_context)

        # Restore reference_id ContextVar from saved attributes
        ref_id = None
        if "attributes" in saved_context and saved_context["attributes"]:
            ref_id = saved_context["attributes"].get("referenceId")

        reference_id_token = reference_id_context.set(ref_id) if ref_id else None
        source_token = uipath_source_context.set(1)

        with trace.use_span(restored_span, end_on_exit=False):
            try:
                yield restored_span
            finally:
                if reference_id_token is not None:
                    reference_id_context.reset(reference_id_token)
                uipath_source_context.reset(source_token)

    async def _handle_suspended(self, agent_span: Span) -> None:
        """Handle suspended state: upsert spans and save trace context.

        Called when agent execution returns SUSPENDED status.
        Upserts pending tool/process spans with UNSET status (no end time)
        so they survive process restart, then saves trace context for resume.

        UNSET status with no end time indicates the span is in-progress/waiting.

        Args:
            agent_span: The current agent span
        """
        tool_name, tool_span, process_span = self._callback.get_pending_tool_info()
        escalation_span = self._callback.get_pending_escalation()
        guardrail_hitl_evaluation_span = (
            self._callback.get_pending_guardrail_hitl_evaluation()
        )
        guardrail_hitl_container_span = (
            self._callback.get_pending_guardrail_hitl_container()
        )
        llm_span = self._callback.get_current_llm()
        current_tool_span = self._callback.get_current_tool()
        resumed_tool_span_data = self._callback.get_resumed_tool_data()

        # Upsert pending spans so they're visible in UI during suspension
        if process_span:
            self._span_factory.upsert_span_suspended(process_span)
        if tool_span:
            self._span_factory.upsert_span_suspended(tool_span)
        self._span_factory.upsert_span_suspended(agent_span)

        if self._trace_context_storage:
            runtime_id = self._get_runtime_id()
            context = self._extract_trace_context(agent_span)

            if tool_name:
                context["pending_tool_name"] = tool_name
                # Save full span data for resume completion
                if tool_span:
                    pending_tool_data = self._extract_pending_span_data(tool_span)
                    context["pending_tool_span"] = pending_tool_data
                    context["pending_tool_span_id"] = pending_tool_data["span_id"]
                if process_span:
                    pending_process_data = self._extract_pending_span_data(process_span)
                    context["pending_process_span"] = pending_process_data
                    context["pending_process_span_id"] = pending_process_data["span_id"]
            # In case we are in a context after a HITL, the current_tool_span is a NonRecordingSpan, but we should have resumed_tool_span_data
            if resumed_tool_span_data:
                context["pending_tool_span"] = resumed_tool_span_data
                context["pending_tool_span_id"] = resumed_tool_span_data["span_id"]

            # Save escalation span data for resume completion
            if escalation_span:
                pending_escalation_data = self._extract_pending_span_data(
                    escalation_span
                )
                context["pending_escalation_span"] = pending_escalation_data

                if guardrail_hitl_evaluation_span:
                    pending_guardrail_hitl_evaluation_data = (
                        self._extract_pending_span_data(guardrail_hitl_evaluation_span)
                    )
                    context["pending_guardrail_hitl_evaluation_span"] = (
                        pending_guardrail_hitl_evaluation_data
                    )

                if guardrail_hitl_container_span:
                    # After a previous resume, the container span in
                    # guardrail_containers is a NonRecordingSpan.
                    # _extract_pending_span_data cannot extract rich data from it.
                    # Prefer the already-stored resumed data when available.
                    resumed_container_data = (
                        self._callback.get_resumed_hitl_guardrail_container_data()
                    )
                    if resumed_container_data:
                        pending_guardrail_hitl_container_data = resumed_container_data
                    else:
                        pending_guardrail_hitl_container_data = (
                            self._extract_pending_span_data(
                                guardrail_hitl_container_span
                            )
                        )
                    context["pending_guardrail_hitl_container_span"] = (
                        pending_guardrail_hitl_container_data
                    )

                if llm_span:
                    # Same pattern as container span: after a previous resume,
                    # current_llm_span may be a NonRecordingSpan. Use resumed
                    # data when available.
                    resumed_llm_data = self._callback.get_resumed_llm_data()
                    if resumed_llm_data:
                        pending_llm_data = resumed_llm_data
                    else:
                        pending_llm_data = self._extract_pending_span_data(llm_span)
                    context["pending_llm_span"] = pending_llm_data

                if resumed_tool_span_data:
                    context["pending_tool_span"] = resumed_tool_span_data
                elif current_tool_span:
                    current_tool_span_data = self._extract_pending_span_data(
                        current_tool_span
                    )
                    context["pending_tool_span"] = current_tool_span_data

            await self._trace_context_storage.save_trace_context(runtime_id, context)
            logger.debug(
                "Saved trace context for runtime %s (trace_id=%s, pending_tool=%s)",
                runtime_id,
                context["trace_id"],
                tool_name,
            )

    def _extract_pending_span_data(self, span: Span) -> PendingSpanData:
        from opentelemetry.sdk.trace import ReadableSpan

        ctx = span.get_span_context()
        data: PendingSpanData = {
            "span_id": format(ctx.span_id, "016x"),
            "parent_span_id": None,
            "name": span.name if hasattr(span, "name") else "unknown",
            "start_time_ns": 0,
            "attributes": {},
        }
        if isinstance(span, ReadableSpan):
            if span.parent and span.parent.span_id:
                data["parent_span_id"] = format(span.parent.span_id, "016x")
            if span.start_time:
                data["start_time_ns"] = span.start_time
            if span.attributes:
                data["attributes"] = dict(span.attributes)
        return data

    def _handle_resume_complete(self, saved_context: TraceContextData) -> None:
        """Complete pending spans on successful resume.

        Called when agent execution completes successfully after resume.
        Upserts pending tool/process spans with OK status and final end_time.

        Note: If callback already completed spans on tool_end (for accurate duration),
        this method becomes a no-op.

        Args:
            saved_context: Previously saved trace context with pending span data
        """
        if self._callback.resumed_spans_completed():
            logger.debug(
                "Resumed spans already completed by callback, skipping duplicate upsert"
            )
            return

        self._complete_pending_tool_span(saved_context)
        self._complete_pending_process_span(saved_context)

        trace_id = saved_context["trace_id"]
        pending_escalation = saved_context.get("pending_escalation_span")
        if pending_escalation:
            pending_escalation["attributes"]["reviewStatus"] = "Completed"
            pending_escalation["attributes"]["reviewOutcome"] = "Approved"
            self._span_factory.upsert_span_complete_by_data(
                trace_id=trace_id,
                span_data=dict(pending_escalation),
            )
            logger.debug(
                "Completed pending escalation span %s",
                pending_escalation.get("name", "unknown"),
            )

    def _handle_resume_error(
        self, saved_context: TraceContextData, error: Exception | UiPathErrorContract
    ) -> None:
        """Complete pending spans with ERROR status on resume failure.

        Args:
            saved_context: Previously saved trace context with pending span data
            error: An exception or error to add to the span
        """
        if self._callback.resumed_spans_completed():
            logger.debug(
                "Resumed spans already completed by callback, skipping duplicate upsert"
            )
            return

        if isinstance(error, UiPathErrorContract):
            error_dict = {
                "message": error.title,
                "detail": error.detail,
            }
        else:
            error_dict = {
                "message": str(error),
                "type": type(error).__name__,
            }
        error_info = json.dumps(error_dict)

        self._complete_pending_tool_span(saved_context, error_info)
        self._complete_pending_process_span(saved_context, error_info)

        trace_id = saved_context["trace_id"]
        pending_escalation = saved_context.get("pending_escalation_span")
        if pending_escalation:
            pending_escalation["attributes"]["error"] = error_info
            self._span_factory.upsert_span_complete_by_data(
                trace_id=trace_id,
                span_data=dict(pending_escalation),
                status=SpanStatus.ERROR,
            )
            logger.debug(
                "Completed pending escalation span %s with ERROR status",
                pending_escalation.get("name", "unknown"),
            )

    def _complete_pending_tool_span(
        self,
        saved_context: TraceContextData,
        error_info: str | None = None,
    ) -> None:
        """Complete a pending tool span. Status is ERROR if error_info is provided."""
        pending_tool = saved_context.get("pending_tool_span")
        if pending_tool:
            trace_id = saved_context["trace_id"]
            if error_info:
                pending_tool["attributes"]["error"] = error_info
                status = SpanStatus.ERROR
            else:
                status = SpanStatus.OK
            self._span_factory.upsert_span_complete_by_data(
                trace_id=trace_id,
                span_data=dict(pending_tool),
                status=status,
            )
            logger.debug(
                "Completed pending tool span %s with %s status",
                pending_tool.get("name", "unknown"),
                "OK" if status == SpanStatus.OK else "ERROR",
            )

    def _complete_pending_process_span(
        self,
        saved_context: TraceContextData,
        error_info: str | None = None,
    ) -> None:
        """Complete a pending process span. Status is ERROR if error_info is provided."""
        pending_process = saved_context.get("pending_process_span")
        if pending_process:
            trace_id = saved_context["trace_id"]
            if error_info:
                pending_process["attributes"]["error"] = error_info
                status = SpanStatus.ERROR
            else:
                status = SpanStatus.OK
            self._span_factory.upsert_span_complete_by_data(
                trace_id=trace_id,
                span_data=dict(pending_process),
                status=status,
            )
            logger.debug(
                "Completed pending process span %s with %s status",
                pending_process.get("name", "unknown"),
                "OK" if status == SpanStatus.OK else "ERROR",
            )

    async def _load_trace_context_if_resuming(self) -> Optional[TraceContextData]:
        if not self._trace_context_storage:
            return None

        runtime_id = self._get_runtime_id()
        saved = await self._trace_context_storage.load_trace_context(runtime_id)

        if saved:
            logger.debug(
                "Loaded saved trace context for runtime %s (trace_id=%s)",
                runtime_id,
                saved["trace_id"],
            )

        return saved

    async def _clear_trace_context(self) -> None:
        if not self._trace_context_storage:
            return

        runtime_id = self._get_runtime_id()
        await self._trace_context_storage.clear_trace_context(runtime_id)
        logger.debug("Cleared trace context for runtime %s", runtime_id)

    def _extract_trace_context(self, span: Span) -> TraceContextData:
        ctx = span.get_span_context()

        # Extract attributes if available (ReadableSpan has attributes)
        attributes: Dict[str, Any] = {}
        if hasattr(span, "attributes") and span.attributes:
            attributes = dict(span.attributes)

        return TraceContextData(
            trace_id=format(ctx.trace_id, "032x"),
            span_id=format(ctx.span_id, "016x"),
            parent_span_id=None,  # Root agent span has no parent
            name=span.name if hasattr(span, "name") else self._get_agent_name(),
            start_time=datetime.now(timezone.utc).isoformat(),
            attributes=attributes,
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name=None,
            pending_tool_span=None,
            pending_process_span=None,
            pending_escalation_span=None,
            pending_guardrail_hitl_evaluation_span=None,
            pending_guardrail_hitl_container_span=None,
            pending_llm_span=None,
        )

    def _get_runtime_id(self) -> str:
        if hasattr(self._delegate, "runtime_id"):
            return self._delegate.runtime_id
        return "unknown"

    def _get_agent_name(self) -> str:
        if self._agent_definition:
            return self._agent_definition.name or "Unknown"
        if hasattr(self._delegate, "runtime_id"):
            return self._delegate.runtime_id
        return "unknown"

    def _build_and_set_enriched_properties(self, agent_span: Span) -> Dict[str, Any]:
        """Build enriched telemetry properties and set them on the callback.

        Shared by both the normal and resume branches so that events tracked
        after an interruption/resume still carry full agent and execution
        metadata.

        Returns:
            The enriched properties dict (also set on the callback).
        """
        agent_name = self._get_agent_name()
        agent_id = self._get_agent_id()

        base_properties: Dict[str, Any] = {
            "AgentName": agent_name,
        }
        if agent_id:
            base_properties["AgentId"] = agent_id

        enriched_properties = self._get_enriched_properties(base_properties, agent_span)
        self._callback.set_enriched_properties(enriched_properties)
        return enriched_properties

    def _get_enriched_properties(
        self, base_properties: Dict[str, Any], agent_span: Optional[Span] = None
    ) -> Dict[str, Any]:
        properties = base_properties.copy()

        trace_id = UiPathConfig.trace_id
        if agent_span:
            span_context = agent_span.get_span_context()
            if span_context and span_context.trace_id:
                trace_id = format(span_context.trace_id, "032x")

        if trace_id:
            properties["TraceId"] = trace_id

        properties["AgentRunId"] = self._agent_run_id

        if self._agent_definition:
            if self._agent_definition.settings.model:
                properties["Model"] = str(self._agent_definition.settings.model)

            if self._agent_definition.settings.max_tokens is not None:
                properties["MaxTokens"] = str(
                    self._agent_definition.settings.max_tokens
                )
            if self._agent_definition.settings.temperature is not None:
                properties["Temperature"] = str(
                    self._agent_definition.settings.temperature
                )

            if self._agent_definition.settings.engine:
                properties["Engine"] = str(self._agent_definition.settings.engine)

            if (
                hasattr(self._agent_definition.settings, "max_iterations")
                and self._agent_definition.settings.max_iterations is not None
            ):
                properties["MaxIterations"] = str(
                    self._agent_definition.settings.max_iterations
                )

            if self._agent_definition.is_conversational is not None:
                properties["IsConversational"] = str(
                    self._agent_definition.is_conversational
                )
            if self._agent_definition.version is not None:
                properties["AgentVersion"] = str(self._agent_definition.version)

        properties["AgentRunSource"] = get_execution_type(self._runtime_context).value
        properties["ApplicationName"] = "UiPath.AgentService"
        properties["Runtime"] = "URT"

        try:
            properties["UiPathAgentsPackageVersion"] = version("uipath-agents")
        except Exception:
            properties["UiPathAgentsPackageVersion"] = "unknown"

        properties["CloudOrganizationId"] = UiPathConfig.organization_id or ""

        properties["CloudTenantId"] = self._runtime_context.tenant_id
        properties["JobId"] = self._runtime_context.job_id

        try:
            cloud_user_id = get_claim_from_token("sub")
            properties["CloudUserId"] = cloud_user_id if cloud_user_id else ""
        except Exception:
            properties["CloudUserId"] = ""

        entrypoint = getattr(self._delegate, "entrypoint", None)
        if entrypoint:
            properties["Entrypoint"] = str(entrypoint)

        properties["AgentType"] = "LowCode"

        if UiPathConfig.folder_key:
            properties["FolderKey"] = UiPathConfig.folder_key
        if UiPathConfig.job_key:
            properties["JobKey"] = UiPathConfig.job_key
        if UiPathConfig.project_id:
            properties["ProjectId"] = UiPathConfig.project_id
        if UiPathConfig.process_uuid:
            properties["ProcessUuid"] = UiPathConfig.process_uuid
        if UiPathConfig.process_version:
            properties["ProcessVersion"] = UiPathConfig.process_version
        properties["ImageVersion"] = os.getenv("IMAGE_VERSION", None)
        return properties

    def _get_agent_id(self) -> Optional[str]:
        if self._agent_definition is not None:
            return self._agent_definition.id
        return None

    def _get_schemas(self) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self._agent_definition:
            return (
                self._agent_definition.input_schema,
                self._agent_definition.output_schema,
            )
        return None, None

    def _get_prompts(self) -> tuple[Optional[str], Optional[str]]:
        if hasattr(self._delegate, "_get_trace_prompts"):
            return self._delegate._get_trace_prompts()

        if self._agent_definition and self._agent_definition.messages:
            from uipath.agent.models.agent import AgentMessageRole

            system_prompt = None
            user_prompt = None
            for msg in self._agent_definition.messages:
                if msg.role == AgentMessageRole.SYSTEM:
                    system_prompt = msg.content
                elif msg.role == AgentMessageRole.USER:
                    user_prompt = msg.content
            return system_prompt, user_prompt
        return None, None

    def _normalize_input(self, input_data: Any) -> Optional[Dict[str, Any]]:
        if isinstance(input_data, dict):
            return input_data
        return None

    def _emit_output_if_successful(
        self,
        result: UiPathRuntimeResult,
        duration_ms: Optional[int] = None,
        agent_span: Optional[Span] = None,
    ) -> None:
        if result.status == UiPathRuntimeStatus.SUCCESSFUL:
            # Set output on parent agentRun span
            if agent_span:
                output = result.output
                if isinstance(output, (dict, list)):
                    output_str = json.dumps(output)
                else:
                    output_str = str(output) if output is not None else ""
                agent_span.set_attribute("output", output_str)

            input_schema, output_schema = self._get_schemas()
            self._span_factory.emit_agent_output(result.output, output_schema)

            if self._event_emitter:
                agent_name = self._get_agent_name()
                agent_id = self._get_agent_id()

                base_properties: Dict[str, Any] = {
                    "AgentName": agent_name,
                    "Status": "Completed",
                    "ErrorMessage": "",
                }

                if agent_id:
                    base_properties["AgentId"] = agent_id

                if duration_ms is not None:
                    base_properties["DurationMs"] = duration_ms

                enriched_properties = self._get_enriched_properties(
                    base_properties, agent_span
                )

                self._event_emitter.track_event(
                    AgentRunEvent.COMPLETED, enriched_properties
                )
