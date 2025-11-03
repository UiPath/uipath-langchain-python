import asyncio
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._debug._bridge import UiPathDebugBridge, get_debug_bridge
from uipath._cli._debug._runtime import UiPathDebugRuntime
from uipath._cli._runtime._contracts import (
    UiPathRuntimeContext,
    UiPathRuntimeFactory,
)
from uipath._cli.middlewares import MiddlewareResult
from uipath.tracing import LlmOpsHttpExporter

from .._tracing import _instrument_traceable_attributes
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._memory import get_memory
from ._runtime._runtime import LangGraphScriptRuntime
from ._utils._graph import LangGraphConfig


def langgraph_debug_middleware(
    entrypoint: Optional[str], input: Optional[str], resume: bool, **kwargs
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:

        async def execute():
            context = UiPathRuntimeContext.with_defaults(**kwargs)
            context.entrypoint = entrypoint
            context.input = input
            context.resume = resume

            _instrument_traceable_attributes()

            async with get_memory(context) as memory:
                runtime_factory = UiPathRuntimeFactory(
                    LangGraphScriptRuntime,
                    UiPathRuntimeContext,
                    runtime_generator=lambda ctx: LangGraphScriptRuntime(
                        ctx, memory, ctx.entrypoint
                    ),
                    context_generator=lambda: context,
                )

                if context.job_id:
                    runtime_factory.add_span_exporter(
                        LlmOpsHttpExporter(extra_process_spans=True)
                    )

                runtime_factory.add_instrumentor(
                    LangChainInstrumentor, get_current_span
                )

                debug_bridge: UiPathDebugBridge = get_debug_bridge(context)

                async with UiPathDebugRuntime.from_debug_context(
                    factory=runtime_factory,
                    context=context,
                    debug_bridge=debug_bridge,
                ) as debug_runtime:
                    await debug_runtime.execute()

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
