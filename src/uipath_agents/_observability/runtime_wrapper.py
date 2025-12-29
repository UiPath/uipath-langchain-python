"""Telemetry wrapper for UiPath runtimes.

Provides telemetry/tracing capabilities via composition (wrapper pattern)
rather than inheritance (mixin pattern).

Manual instrumentation is always enabled for dual instrumentation:
- Manual spans (agentRun, llmCall, toolCall) → LLMOps (user-facing)
- OpenInference spans → AppInsights (debugging)
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

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
from .tracer import UiPathTracer

logger = logging.getLogger(__name__)


class TelemetryRuntimeWrapper:
    """Wrapper that adds telemetry to any UiPathRuntimeProtocol implementation.

    Uses composition pattern to wrap any runtime with tracing capabilities.
    Manages agent span lifecycle and updates the callback before each execution.

    The callback is passed to the delegate runtime via constructor (not context vars),
    ensuring it persists across debug/chat re-executions where the same runtime
    instance is executed multiple times.

    Example:
        tracer = UiPathTracer()
        callback = UiPathTracingCallback(tracer)
        base_runtime = AgentsLangGraphRuntime(graph, callbacks=[callback])
        traced_runtime = TelemetryRuntimeWrapper(base_runtime, tracer, callback)
        result = await traced_runtime.execute(input_data)
    """

    def __init__(
        self,
        delegate: UiPathRuntimeProtocol,
        tracer: UiPathTracer,
        callback: UiPathTracingCallback,
        agent_info: Optional[AgentSpanInfo] = None,
    ):
        self._delegate = delegate
        self._tracer = tracer
        self._callback = callback
        self._agent_info = agent_info

    @property
    def delegate(self) -> UiPathRuntimeProtocol:
        return self._delegate

    async def execute(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        async with self._agent_span_context(input):
            result = await self._delegate.execute(input, options)
            self._emit_output_if_successful(result)
            return result

    async def stream(
        self,
        input: Dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        async with self._agent_span_context(input):
            final_result: Optional[UiPathRuntimeResult] = None

            async for event in self._delegate.stream(input, options):
                if isinstance(event, UiPathRuntimeResult):
                    final_result = event
                yield event

            if final_result:
                self._emit_output_if_successful(final_result)

    async def get_schema(self) -> UiPathRuntimeSchema:
        return await self._delegate.get_schema()

    async def dispose(self) -> None:
        await self._delegate.dispose()

    @asynccontextmanager
    async def _agent_span_context(
        self, input_data: Any = None
    ) -> AsyncGenerator[None, None]:
        """Context manager for agent span lifecycle.

        Creates agent span and updates the callback to use it as root.
        The callback was already passed to delegate via constructor,
        so it will automatically receive events during execution.
        """
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
                yield
            finally:
                self._callback.cleanup()

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
