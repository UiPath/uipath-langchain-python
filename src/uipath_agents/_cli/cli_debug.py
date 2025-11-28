import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
from uipath._cli._debug._bridge import get_debug_bridge
from uipath._cli._utils._studio_project import StudioClient
from uipath._cli.middlewares import MiddlewareResult
from uipath._config import UiPathConfig
from uipath._utils._bindings import ResourceOverwritesContext
from uipath.core import UiPathTraceManager
from uipath.runtime import (
    UiPathDebugBridgeProtocol,
    UiPathDebugRuntime,
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
    UiPathRuntimeProtocol,
)
from uipath.tracing import LlmOpsHttpExporter
from uipath_langchain._cli._runtime._exception import LangGraphRuntimeError

from .._observability import shutdown_telemetry
from .utils import _prepare_agent_run_files

load_dotenv()

logger = logging.getLogger(__name__)


def agents_debug_middleware(
    entrypoint: Optional[str],
    input: Optional[str],
    resume: bool,
    input_file: Optional[str],
    output_file: Optional[str],
    **kwargs,
) -> MiddlewareResult:
    """Middleware to handle Agents LangGraph execution"""
    _prepare_agent_run_files()

    try:

        async def execute_debug_runtime():
            trace_manager = UiPathTraceManager()

            with UiPathRuntimeContext.with_defaults(
                entrypoint=entrypoint,
                input=input,
                input_file=input_file,
                output_file=output_file,
                resume=resume,
                trace_manager=trace_manager,
                command="debug",
            ) as ctx:
                runtime: UiPathRuntimeProtocol | None = None
                debug_runtime: UiPathRuntimeProtocol | None = None
                factory: UiPathRuntimeFactoryProtocol | None = None

                try:
                    if ctx.job_id:
                        trace_manager.add_span_exporter(LlmOpsHttpExporter())

                    factory = UiPathRuntimeFactoryRegistry.get(context=ctx)

                    runtime = await factory.new_runtime(
                        ctx.entrypoint, ctx.job_id or "agents"
                    )

                    debug_bridge: UiPathDebugBridgeProtocol = get_debug_bridge(ctx)

                    debug_runtime = UiPathDebugRuntime(
                        delegate=runtime,
                        debug_bridge=debug_bridge,
                    )

                    project_id = UiPathConfig.project_id

                    if project_id:
                        studio_client = StudioClient(project_id)

                        async with ResourceOverwritesContext(
                            lambda: studio_client.get_resource_overwrites()
                        ):
                            ctx.result = await debug_runtime.execute(
                                ctx.get_input(),
                                options=UiPathExecuteOptions(resume=resume),
                            )
                    else:
                        ctx.result = await debug_runtime.execute(
                            ctx.get_input(),
                            options=UiPathExecuteOptions(resume=resume),
                        )

                finally:
                    if debug_runtime:
                        await debug_runtime.dispose()
                    if runtime:
                        await runtime.dispose()
                    if factory:
                        await factory.dispose()

        asyncio.run(execute_debug_runtime())

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
    finally:
        try:
            shutdown_telemetry()
        except Exception as e:
            logger.error(f"Error during telemetry shutdown: {e}")
