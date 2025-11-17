import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._debug._bridge import UiPathDebugBridge, get_debug_bridge
from uipath._cli._debug._runtime import UiPathDebugRuntime
from uipath._cli._runtime._contracts import UiPathRuntimeContext
from uipath._cli._utils._studio_project import StudioClient
from uipath._cli.middlewares import MiddlewareResult
from uipath._config import UiPathConfig
from uipath._utils._bindings import ResourceOverwritesContext
from uipath.tracing import LlmOpsHttpExporter
from uipath_langchain._cli._runtime._exception import LangGraphRuntimeError
from uipath_langchain._cli._runtime._memory import get_memory
from uipath_langchain._tracing import _instrument_traceable_attributes

from .._observability import get_azure_exporter, shutdown_telemetry
from .constants import BINDINGS_FILENAME, ROOT_BINDINGS_FILENAME
from .runtime import create_agent_langgraph_runtime, setup_runtime_factory

load_dotenv()

logger = logging.getLogger(__name__)


def _prepare_bindings_file() -> None:
    """Copy bindings.json from .agent-builder/ to root directory if it exists."""
    source_path = Path(BINDINGS_FILENAME)
    target_path = Path(ROOT_BINDINGS_FILENAME)

    if not source_path.exists():
        logger.debug(f"Source bindings file not found at {source_path}")
        return

    try:
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied bindings.json from {source_path} to {target_path}")
    except Exception as e:
        logger.error(f"Failed to copy bindings.json: {e}")
        raise


def lowcode_debug_middleware(
    entrypoint: Optional[str], input: Optional[str], resume: bool, **kwargs
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
                # Set up tracing instrumentation
                _instrument_traceable_attributes()

                runtime_factory = setup_runtime_factory(
                    runtime_generator=lambda ctx: create_agent_langgraph_runtime(
                        ctx, memory
                    ),
                    context_generator=lambda: context,
                )
                runtime_factory.add_instrumentor(
                    LangChainInstrumentor, get_current_span
                )

                if context.job_id:
                    runtime_factory.add_span_exporter(
                        LlmOpsHttpExporter(extra_process_spans=True)
                    )

                azure_exporter = get_azure_exporter()
                if azure_exporter:
                    runtime_factory.add_span_exporter(azure_exporter)

                async def execute_debug_runtime():
                    async with UiPathDebugRuntime.from_debug_context(
                        factory=runtime_factory,
                        context=context,
                        debug_bridge=debug_bridge,
                    ) as debug_runtime:
                        await debug_runtime.execute()

                debug_bridge: UiPathDebugBridge = get_debug_bridge(context)
                project_id = UiPathConfig.project_id

                _prepare_bindings_file()

                if project_id:
                    studio_client = StudioClient(project_id)

                    async with ResourceOverwritesContext(
                        lambda: studio_client.get_resource_overwrites()
                    ) as ctx:
                        logger.info(
                            f"Loaded {ctx.overwrites_count} resource overwrites"
                        )
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
    finally:
        try:
            shutdown_telemetry()
        except Exception as e:
            logger.error(f"Error during telemetry shutdown: {e}")
