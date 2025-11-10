import asyncio
import logging
from typing import Optional

from uipath._cli._debug._bridge import ConsoleDebugBridge, UiPathDebugBridge
from uipath._cli._runtime._contracts import UiPathRuntimeContext, UiPathRuntimeResult
from uipath._cli._utils._common import read_resource_overwrites_from_file
from uipath._cli.middlewares import MiddlewareResult
from uipath._events._events import UiPathAgentStateEvent
from uipath._utils._bindings import ResourceOverwritesContext
from uipath.tracing import JsonLinesFileExporter, LlmOpsHttpExporter
from uipath_langchain._cli._runtime._exception import LangGraphRuntimeError
from uipath_langchain._cli._runtime._memory import get_memory

from .._observability import get_azure_exporter, shutdown_telemetry
from .runtime import create_agent_langgraph_runtime, setup_runtime_factory

logger = logging.getLogger(__name__)


def lowcode_run_middleware(
    entrypoint: Optional[str],
    input: Optional[str],
    resume: bool,
    trace_file: Optional[str] = None,
    **kwargs,
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""
    try:
        context = UiPathRuntimeContext.with_defaults(**kwargs)
        context.entrypoint = entrypoint
        context.input = input
        context.resume = resume
        context.execution_id = context.job_id or "default"

        async def execute():
            async with get_memory(context) as memory:
                runtime_factory = setup_runtime_factory(
                    runtime_generator=lambda ctx: create_agent_langgraph_runtime(
                        ctx, memory
                    )
                )

                if trace_file:
                    runtime_factory.add_span_exporter(JsonLinesFileExporter(trace_file))

                if context.job_id:
                    async with ResourceOverwritesContext(
                        lambda: read_resource_overwrites_from_file(context.runtime_dir)
                    ):
                        runtime_factory.add_span_exporter(
                            LlmOpsHttpExporter(extra_process_spans=True)
                        )

                        azure_exporter = get_azure_exporter()
                        if azure_exporter:
                            runtime_factory.add_span_exporter(azure_exporter)

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
    finally:
        try:
            shutdown_telemetry()
        except Exception as e:
            logger.error(f"Error during telemetry shutdown: {e}")
