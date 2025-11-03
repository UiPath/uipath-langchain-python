import asyncio
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._debug._bridge import ConsoleDebugBridge, UiPathDebugBridge
from uipath._cli._runtime._contracts import (
    UiPathRuntimeContext,
    UiPathRuntimeFactory,
    UiPathRuntimeResult,
)
from uipath._cli.middlewares import MiddlewareResult
from uipath._events._events import UiPathAgentStateEvent
from uipath.tracing import JsonLinesFileExporter, LlmOpsHttpExporter

from .._tracing import (
    _instrument_traceable_attributes,
)
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._memory import get_memory
from ._runtime._runtime import LangGraphScriptRuntime
from ._utils._graph import LangGraphConfig


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
        _instrument_traceable_attributes()

        async def execute():
            context = UiPathRuntimeContext.with_defaults(**kwargs)
            context.entrypoint = entrypoint
            context.input = input
            context.resume = resume

            async with get_memory(context) as memory:
                runtime_factory = UiPathRuntimeFactory(
                    LangGraphScriptRuntime,
                    UiPathRuntimeContext,
                    runtime_generator=lambda ctx: LangGraphScriptRuntime(
                        ctx, memory, ctx.entrypoint
                    ),
                )

                runtime_factory.add_instrumentor(
                    LangChainInstrumentor, get_current_span
                )

                if trace_file:
                    runtime_factory.add_span_exporter(JsonLinesFileExporter(trace_file))

                if context.job_id:
                    runtime_factory.add_span_exporter(
                        LlmOpsHttpExporter(extra_process_spans=True)
                    )
                    await runtime_factory.execute(context)
                else:
                    debug_bridge: UiPathDebugBridge = ConsoleDebugBridge()
                    await debug_bridge.emit_execution_started("default")
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
