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

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanContext, TraceFlags
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeProtocol,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.events import UiPathRuntimeEvent
from uipath.runtime.schema import UiPathRuntimeSchema

from .callback import UiPathTracingCallback
from .span_attributes import AgentSpanInfo
from .trace_context_storage import TraceContextData, TraceContextStorage
from .tracer import UiPathTracer

logger = logging.getLogger(__name__)


class TelemetryRuntimeWrapper:
    """Wrapper that adds telemetry to any UiPathRuntimeProtocol implementation.

    Uses composition pattern to wrap any runtime with tracing capabilities.
    Manages agent span lifecycle and updates the callback before each execution.

    The callback is passed to the delegate runtime via constructor (not context vars),
    ensuring it persists across debug/chat re-executions where the same runtime
    instance is executed multiple times.

    Supports trace context preservation across suspend/resume cycles:
    - On SUSPENDED: upserts span with RUNNING status and persists trace context
    - On resume: loads trace context and continues the same trace

    Example:
        tracer = UiPathTracer()
        callback = UiPathTracingCallback(tracer)
        storage = SqliteTraceContextStorage(sqlite_storage)
        base_runtime = AgentsLangGraphRuntime(graph, callbacks=[callback])
        traced_runtime = TelemetryRuntimeWrapper(
            base_runtime, tracer, callback, trace_context_storage=storage
        )
        result = await traced_runtime.execute(input_data)
    """

    def __init__(
        self,
        delegate: UiPathRuntimeProtocol,
        tracer: UiPathTracer,
        callback: UiPathTracingCallback,
        agent_info: Optional[AgentSpanInfo] = None,
        trace_context_storage: Optional[TraceContextStorage] = None,
    ):
        """Initialize the telemetry wrapper.

        Args:
            delegate: The runtime to wrap with telemetry
            tracer: UiPathTracer for creating spans
            callback: Callback for LangChain event instrumentation
            agent_info: Optional agent metadata for span attributes
            trace_context_storage: Optional storage for trace context preservation
                across suspend/resume cycles. If None, trace context is not preserved.
        """
        self._delegate = delegate
        self._tracer = tracer
        self._callback = callback
        self._agent_info = agent_info
        self._trace_context_storage = trace_context_storage

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
        # Check for existing trace context (resume scenario)
        saved_context = await self._load_trace_context_if_resuming()

        async with self._agent_span_context(input, saved_context) as agent_span:
            result = await self._delegate.execute(input, options)

            if result.status == UiPathRuntimeStatus.SUSPENDED:
                await self._handle_suspended(agent_span)
            elif result.status == UiPathRuntimeStatus.SUCCESSFUL:
                self._emit_output_if_successful(result)
                await self._clear_trace_context()
            elif result.status == UiPathRuntimeStatus.FAULTED:
                await self._clear_trace_context()

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
        saved_context = await self._load_trace_context_if_resuming()

        async with self._agent_span_context(input, saved_context) as agent_span:
            final_result: Optional[UiPathRuntimeResult] = None

            async for event in self._delegate.stream(input, options):
                if isinstance(event, UiPathRuntimeResult):
                    final_result = event
                yield event

            if final_result:
                if final_result.status == UiPathRuntimeStatus.SUSPENDED:
                    await self._handle_suspended(agent_span)
                elif final_result.status == UiPathRuntimeStatus.SUCCESSFUL:
                    self._emit_output_if_successful(final_result)
                    await self._clear_trace_context()
                elif final_result.status == UiPathRuntimeStatus.FAULTED:
                    await self._clear_trace_context()

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
    ) -> AsyncGenerator[Span, None]:
        """Context manager for agent span lifecycle.

        Creates or restores agent span and updates the callback to use it as root.
        The callback was already passed to delegate via constructor,
        so it will automatically receive events during execution.

        Args:
            input_data: Input data for the agent (used for new spans)
            saved_context: Previously saved trace context (resume scenario)

        Yields:
            The agent span for the execution
        """
        if saved_context:
            # Resume: restore original trace context
            async with self._restore_trace_context(saved_context) as agent_span:
                self._callback.set_agent_span(agent_span)
                # Tell callback to skip span creation for the pending tool
                pending_tool = saved_context.get("pending_tool_name")
                if pending_tool:
                    self._callback.set_resume_context(pending_tool)
                try:
                    yield agent_span
                finally:
                    self._callback.cleanup()
        else:
            # Normal: create new trace context
            agent_name = self._get_agent_name()
            system_prompt, user_prompt = self._get_prompts()
            input_schema, output_schema = self._get_schemas()

            with self._tracer.start_agent_run(
                agent_name=agent_name,
                agent_id=self._get_agent_id(),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                input_data=self._normalize_input(input_data),
                input_schema=input_schema,
                output_schema=output_schema,
            ) as agent_span:
                self._callback.set_agent_span(agent_span)
                try:
                    yield agent_span
                finally:
                    self._callback.cleanup()

    @asynccontextmanager
    async def _restore_trace_context(
        self, saved_context: TraceContextData
    ) -> AsyncGenerator[Span, None]:
        """Restore a previously saved trace context for resume.

        Restores the original agent span context so new spans created
        during resume become children of the original agent span (not
        nested under a new child span).

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

        with trace.use_span(restored_span, end_on_exit=False):
            try:
                yield restored_span
            except Exception:
                raise

    async def _handle_suspended(self, agent_span: Span) -> None:
        """Handle suspended state: save trace context for re-parenting.

        Called when agent execution returns SUSPENDED status.
        Saves trace context so resumed execution can link to original trace.
        Also saves pending tool span info to avoid duplicate spans on resume.

        Args:
            agent_span: The current agent span
        """
        if self._trace_context_storage:
            runtime_id = self._get_runtime_id()
            context = self._extract_trace_context(agent_span)

            # Get pending tool span info from callback
            tool_name, tool_span, process_span = self._callback.get_pending_tool_info()
            if tool_name:
                context["pending_tool_name"] = tool_name
                if tool_span:
                    ctx = tool_span.get_span_context()
                    context["pending_tool_span_id"] = format(ctx.span_id, "016x")
                if process_span:
                    ctx = process_span.get_span_context()
                    context["pending_process_span_id"] = format(ctx.span_id, "016x")

            await self._trace_context_storage.save_trace_context(runtime_id, context)
            logger.debug(
                "Saved trace context for runtime %s (trace_id=%s, pending_tool=%s)",
                runtime_id,
                context["trace_id"],
                tool_name,
            )

    async def _load_trace_context_if_resuming(self) -> Optional[TraceContextData]:
        """Load trace context if this is a resume execution.

        Checks storage for previously saved trace context.

        Returns:
            Saved trace context if exists, None otherwise
        """
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
        """Clear trace context after completion.

        Called on terminal states (SUCCESSFUL, FAILED) to clean up
        saved trace context.
        """
        if not self._trace_context_storage:
            return

        runtime_id = self._get_runtime_id()
        await self._trace_context_storage.clear_trace_context(runtime_id)
        logger.debug("Cleared trace context for runtime %s", runtime_id)

    def _extract_trace_context(self, span: Span) -> TraceContextData:
        """Extract trace context data from a span for persistence.

        Args:
            span: The span to extract context from

        Returns:
            TraceContextData for persistence
        """
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
        )

    def _get_runtime_id(self) -> str:
        if hasattr(self._delegate, "runtime_id"):
            return self._delegate.runtime_id
        return "unknown"

    def _get_agent_name(self) -> str:
        if self._agent_info:
            return self._agent_info.name
        if hasattr(self._delegate, "runtime_id"):
            return self._delegate.runtime_id
        return "unknown"

    def _get_agent_id(self) -> Optional[str]:
        if hasattr(self._delegate, "runtime_id"):
            return self._delegate.runtime_id
        return None

    def _get_schemas(self) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self._agent_info:
            return self._agent_info.input_schema, self._agent_info.output_schema
        return None, None

    def _get_prompts(self) -> tuple[Optional[str], Optional[str]]:
        if hasattr(self._delegate, "_get_trace_prompts"):
            return self._delegate._get_trace_prompts()
        return None, None

    def _normalize_input(self, input_data: Any) -> Optional[Dict[str, Any]]:
        if isinstance(input_data, dict):
            return input_data
        return None

    def _emit_output_if_successful(self, result: UiPathRuntimeResult) -> None:
        if result.status == UiPathRuntimeStatus.SUCCESSFUL:
            self._tracer.emit_agent_output(result.output)
