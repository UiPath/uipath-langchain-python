import asyncio
import os
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._runtime._contracts import (
    UiPathRuntimeFactory,
)
from uipath._cli.middlewares import MiddlewareResult

from .._tracing import LangChainExporter, _instrument_traceable_attributes
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._runtime import (  # type: ignore[attr-defined]
    LangGraphRuntime,
    LangGraphRuntimeContext,
)
from ._utils._graph import LangGraphConfig


def langgraph_run_middleware(
    entrypoint: Optional[str], input: Optional[str], resume: bool, **kwargs
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        context = LangGraphRuntimeContext.with_defaults(**kwargs)
        context.langgraph_config = config
        context.entrypoint = entrypoint
        context.input = input
        context.resume = resume

        _instrument_traceable_attributes()

        def generate_runtime(ctx: LangGraphRuntimeContext) -> LangGraphRuntime:
            runtime = LangGraphRuntime(ctx)
            # If not resuming and no job id, delete the previous state file
            if not ctx.resume and ctx.job_id is None:
                if os.path.exists(runtime.state_file_path):
                    os.remove(runtime.state_file_path)
            return runtime

        async def execute():
            runtime_factory = UiPathRuntimeFactory(
                LangGraphRuntime,
                LangGraphRuntimeContext,
                runtime_generator=generate_runtime,
            )

            if context.job_id:
                runtime_factory.add_span_exporter(LangChainExporter())

            runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)

            await runtime_factory.execute(context)

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
