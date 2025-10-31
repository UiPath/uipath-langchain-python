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
from uipath._cli._utils._common import read_resource_overwrites_from_file
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli.middlewares import MiddlewareResult
from uipath._events._events import UiPathAgentStateEvent
from uipath._utils._bindings import ResourceOverwritesContext
from uipath.tracing import JsonLinesFileExporter, LlmOpsHttpExporter

from .._tracing import (
    _instrument_traceable_attributes,
)
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._runtime import (  # type: ignore[attr-defined]
    LangGraphRuntimeContext,
    LangGraphScriptRuntime,
)
from ._utils._graph import LangGraphConfig

console = ConsoleLogger.get_instance()

def langgraph_run_middleware(
    entrypoint: Optional[str],
    input: Optional[str],
    resume: bool,
    trace_file: Optional[str] = None,
    **kwargs,
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:

        async def execute():
            context = LangGraphRuntimeContext.with_defaults(**kwargs)
            context.entrypoint = entrypoint
            context.input = input
            context.resume = resume
            context.execution_id = context.job_id or "default"
            _instrument_traceable_attributes()

            def generate_runtime(
                ctx: LangGraphRuntimeContext,
            ) -> LangGraphScriptRuntime:
                runtime = LangGraphScriptRuntime(ctx, ctx.entrypoint)
                # If not resuming and no job id, delete the previous state file
                if not ctx.resume and ctx.job_id is None:
                    if os.path.exists(runtime.state_file_path):
                        os.remove(runtime.state_file_path)
                return runtime

            runtime_factory = UiPathRuntimeFactory(
                LangGraphScriptRuntime,
                LangGraphRuntimeContext,
                runtime_generator=generate_runtime,
            )

            runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)

            if trace_file:
                runtime_factory.add_span_exporter(JsonLinesFileExporter(trace_file))

            if context.job_id:
                async with ResourceOverwritesContext(
                    lambda: read_resource_overwrites_from_file(context.runtime_dir)
                ) as ctx:
                    console.info(
                        f"Applied {ctx.overwrites_count} resource overwrite(s)"
                    )
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
