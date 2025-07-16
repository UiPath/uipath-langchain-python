import asyncio
from os import environ as env
from typing import Optional

from dotenv import load_dotenv
from uipath._cli._runtime._contracts import UiPathTraceContext
from uipath._cli.middlewares import MiddlewareResult

from ._runtime._context import LangGraphRuntimeContext
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._runtime import LangGraphRuntime
from ._utils._graph import LangGraphConfig

load_dotenv()


def langgraph_run_middleware(
    entrypoint: Optional[str], input: Optional[str], resume: bool
) -> MiddlewareResult:
    """Middleware to handle langgraph execution"""
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        bool_map = {"true": True, "false": False}
        tracing = env.get("UIPATH_TRACING_ENABLED", True)
        if isinstance(tracing, str) and tracing.lower() in bool_map:
            tracing = bool_map[tracing.lower()]

        async def execute():
            context = LangGraphRuntimeContext.from_config(
                env.get("UIPATH_CONFIG_PATH", "uipath.json")
            )
            config_path = env.get("UIPATH_CONFIG_PATH", "uipath.json")
            console.info(f"[DEBUG] Using config path: {config_path}")
            console.info(f"[DEBUG] Entrypoint argument: {entrypoint}")
            console.info(f"[DEBUG] Entrypoint absolute path: {os.path.abspath(entrypoint) if entrypoint else None}")
            console.info(f"[DEBUG] Entrypoint exists: {os.path.exists(entrypoint) if entrypoint else None}")
            console.info(f"[DEBUG] Input: {input}")
            
            context.entrypoint = entrypoint
            context.input = input
            context.resume = resume
            context.langgraph_config = config
            context.logs_min_level = env.get("LOG_LEVEL", "INFO")
            context.job_id = env.get("UIPATH_JOB_KEY")
            context.trace_id = env.get("UIPATH_TRACE_ID")
            context.tracing_enabled = tracing
            context.trace_context = UiPathTraceContext(
                enabled=tracing,
                trace_id=env.get("UIPATH_TRACE_ID"),
                parent_span_id=env.get("UIPATH_PARENT_SPAN_ID"),
                root_span_id=env.get("UIPATH_ROOT_SPAN_ID"),
                job_id=env.get("UIPATH_JOB_KEY"),
                org_id=env.get("UIPATH_ORGANIZATION_ID"),
                tenant_id=env.get("UIPATH_TENANT_ID"),
                process_key=env.get("UIPATH_PROCESS_UUID"),
                folder_key=env.get("UIPATH_FOLDER_KEY"),
            )
            context.langsmith_tracing_enabled = env.get("LANGSMITH_TRACING", False)

            # Add default env variables
            env["UIPATH_REQUESTING_PRODUCT"] = "uipath-python-sdk"
            env["UIPATH_REQUESTING_FEATURE"] = "langgraph-agent"

            async with LangGraphRuntime.from_context(context) as runtime:
                await runtime.execute()

        asyncio.run(execute())

        return MiddlewareResult(should_continue=False, error_message=None)

    except LangGraphRuntimeError as e:
        console.info(f"[DEBUG][UiPathRuntimeError] {type(e).__name__}: {e}")
        return MiddlewareResult(
            should_continue=False,
            error_message=e.error_info.detail,
            should_include_stacktrace=True,
        )
    except Exception as e:
        console.info(f"[DEBUG][Exception] {type(e).__name__}: {e}")
        return MiddlewareResult(
            should_continue=False,
            error_message=f"Error: {str(e)}",
            should_include_stacktrace=True,
        )
