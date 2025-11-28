import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
from uipath._cli._debug._bridge import ConsoleDebugBridge
from uipath._cli._utils._common import read_resource_overwrites_from_file
from uipath._cli.middlewares import MiddlewareResult
from uipath._utils._bindings import ResourceOverwritesContext
from uipath.core import UiPathTraceManager
from uipath.runtime import (
    UiPathDebugBridgeProtocol,
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
    UiPathRuntimeProtocol,
    UiPathRuntimeResult,
    UiPathStreamOptions,
)
from uipath.runtime.events import UiPathRuntimeStateEvent
from uipath.tracing import JsonLinesFileExporter, LlmOpsHttpExporter
from uipath_langchain._cli._runtime._exception import LangGraphRuntimeError

from .utils import _prepare_agent_run_files

load_dotenv()

logger = logging.getLogger(__name__)


async def execute_runtime(ctx: UiPathRuntimeContext) -> UiPathRuntimeResult:
    with ctx:
        runtime: UiPathRuntimeProtocol | None = None
        factory: UiPathRuntimeFactoryProtocol | None = None
        try:
            if ctx.entrypoint is None:
                raise ValueError("Entrypoint is required for runtime execution")

            factory = UiPathRuntimeFactoryRegistry.get(context=ctx)
            runtime = await factory.new_runtime(ctx.entrypoint, ctx.job_id or "default")
            options = UiPathExecuteOptions(resume=ctx.resume)
            ctx.result = await runtime.execute(input=ctx.get_input(), options=options)
            return ctx.result
        finally:
            if runtime:
                await runtime.dispose()
            if factory:
                await factory.dispose()


async def debug_runtime(
    ctx: UiPathRuntimeContext,
) -> UiPathRuntimeResult | None:
    with ctx:
        runtime: UiPathRuntimeProtocol | None = None
        factory: UiPathRuntimeFactoryProtocol | None = None
        try:
            if ctx.entrypoint is None:
                raise ValueError("Entrypoint is required for runtime debugging")
            factory = UiPathRuntimeFactoryRegistry.get(context=ctx)
            runtime = await factory.new_runtime(ctx.entrypoint, "agents")
            debug_bridge: UiPathDebugBridgeProtocol = ConsoleDebugBridge()
            await debug_bridge.emit_execution_started()
            options = UiPathStreamOptions(resume=ctx.resume)
            async for event in runtime.stream(ctx.get_input(), options=options):
                if isinstance(event, UiPathRuntimeResult):
                    await debug_bridge.emit_execution_completed(event)
                    ctx.result = event
                elif isinstance(event, UiPathRuntimeStateEvent):
                    await debug_bridge.emit_state_update(event)
            return ctx.result
        finally:
            if runtime:
                await runtime.dispose()
            if factory:
                await factory.dispose()


def agents_run_middleware(
    entrypoint: Optional[str],
    input: Optional[str],
    resume: bool,
    input_file: Optional[str],
    output_file: Optional[str],
    trace_file: Optional[str],
    **kwargs,
) -> MiddlewareResult:
    """Middleware to handle Agents LangGraph execution"""
    _prepare_agent_run_files()

    try:

        async def execute() -> None:
            trace_manager = UiPathTraceManager()

            ctx = UiPathRuntimeContext.with_defaults(
                entrypoint=entrypoint,
                input=input,
                input_file=input_file,
                output_file=output_file,
                trace_file=trace_file,
                resume=resume,
                command="run",
                trace_manager=trace_manager,
            )

            if ctx.trace_file:
                trace_manager.add_span_exporter(JsonLinesFileExporter(ctx.trace_file))

            if ctx.job_id:
                trace_manager.add_span_exporter(LlmOpsHttpExporter())

                async with ResourceOverwritesContext(
                    lambda: read_resource_overwrites_from_file(ctx.runtime_dir)
                ) as rcs_ctx:
                    logger.info(
                        f"Applied {rcs_ctx.overwrites_count} resource overwrite(s)"
                    )
                    await execute_runtime(ctx)

            else:
                await debug_runtime(ctx)

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
