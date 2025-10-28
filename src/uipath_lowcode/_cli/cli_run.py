import asyncio
import os
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._debug._bridge import ConsoleDebugBridge, UiPathDebugBridge
from uipath._cli._runtime._contracts import (
    UiPathRuntimeFactory,
    UiPathRuntimeResult,
)
from uipath._cli.middlewares import MiddlewareResult
from uipath._events._events import UiPathAgentStateEvent
from uipath.tracing import JsonLinesFileExporter, LlmOpsHttpExporter
from uipath_langchain._cli._runtime._context import (
    LangGraphRuntimeContext,
)
from uipath_langchain._cli._runtime._exception import (
    LangGraphRuntimeError,
)
from uipath_langchain._cli._runtime._runtime import (
    LangGraphRuntime,
    LangGraphScriptRuntime,
)
from uipath_langchain._tracing import (
    _instrument_traceable_attributes,
)

from ..lowcode_agent_graph_builder import build_lowcode_agent_graph


def lowcode_run_middleware(
    entrypoint: Optional[str],
    input: Optional[str],
    resume: bool,
    trace_file: Optional[str] = None,
    **kwargs,
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""

    try:
        context = LangGraphRuntimeContext.with_defaults(**kwargs)
        context.entrypoint = entrypoint
        context.input = input
        context.resume = resume
        context.execution_id = context.job_id or "default"

        _instrument_traceable_attributes()

        def generate_runtime(ctx: LangGraphRuntimeContext) -> LangGraphRuntime:
            async def graph_builder():
                return await build_lowcode_agent_graph(input_data=ctx.input)

            runtime = LangGraphRuntime(ctx, graph_builder)
            # If not resuming and no job id, delete the previous state file
            if not ctx.resume and ctx.job_id is None:
                if os.path.exists(runtime.state_file_path):
                    os.remove(runtime.state_file_path)
            return runtime

        async def execute():
            runtime_factory = UiPathRuntimeFactory(
                LangGraphScriptRuntime,
                LangGraphRuntimeContext,
                runtime_generator=generate_runtime,
            )

            runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)

            if trace_file:
                runtime_factory.add_span_exporter(JsonLinesFileExporter(trace_file))

            if context.job_id:
                runtime_factory.add_span_exporter(
                    LlmOpsHttpExporter(extra_process_spans=True)
                )
                await runtime_factory.execute(context)
            else:
                debug_bridge: UiPathDebugBridge = ConsoleDebugBridge()
                await debug_bridge.emit_execution_started(context.execution_id)
                async for event in runtime_factory.stream(context):
                    if isinstance(event, UiPathRuntimeResult):
                        await debug_bridge.emit_execution_completed(event)
                    elif isinstance(event, UiPathAgentStateEvent):
                        await debug_bridge.emit_state_update(event)

        asyncio.run(execute())

        return MiddlewareResult(
            should_continue=False,
            error_message=None,
        )

    except LangGraphRuntimeError as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=e.error_info.detail,
            should_include_stacktrace=True,
        )
    except Exception as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=f"Error: {str(e)}",
            should_include_stacktrace=True,
        )
