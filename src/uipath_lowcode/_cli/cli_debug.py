import asyncio
from typing import Optional

from uipath._cli._debug._bridge import UiPathDebugBridge, get_debug_bridge
from uipath._cli._debug._runtime import UiPathDebugRuntime
from uipath._cli._utils._studio_project import StudioClient
from uipath._cli.middlewares import MiddlewareResult
from uipath._config import UiPathConfig
from uipath._utils._bindings import ResourceOverwritesContext
from uipath.tracing import LlmOpsHttpExporter
from uipath_langchain._cli._runtime._context import LangGraphRuntimeContext
from uipath_langchain._cli._runtime._exception import LangGraphRuntimeError

from .runtime import create_agent_langgraph_runtime, setup_runtime_factory


def lowcode_debug_middleware(
    entrypoint: Optional[str], input: Optional[str], resume: bool, **kwargs
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""
    try:
        context = LangGraphRuntimeContext.with_defaults(**kwargs)
        context.entrypoint = entrypoint
        context.input = input
        context.resume = resume
        context.execution_id = context.job_id or "default"

        async def execute():
            runtime_factory = setup_runtime_factory(
                runtime_generator=create_agent_langgraph_runtime,
                context_generator=lambda: context,
            )

            if context.job_id:
                runtime_factory.add_span_exporter(
                    LlmOpsHttpExporter(extra_process_spans=True)
                )

            async def execute_debug_runtime():
                async with UiPathDebugRuntime.from_debug_context(
                    factory=runtime_factory,
                    context=context,
                    debug_bridge=debug_bridge,
                ) as debug_runtime:
                    await debug_runtime.execute()

            debug_bridge: UiPathDebugBridge = get_debug_bridge(context)
            project_id = UiPathConfig.project_id

            if project_id:
                studio_client = StudioClient(project_id)

                async with ResourceOverwritesContext(
                    lambda: studio_client.get_resource_overwrites()
                ) as ctx:
                    print(ctx.overwrites_count)
                    await execute_debug_runtime()
            else:
                await execute_debug_runtime()

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
