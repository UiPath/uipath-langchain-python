import asyncio
import os
from os import environ as env
from typing import Optional

from dotenv import load_dotenv
from uipath._cli.middlewares import MiddlewareResult

from ._runtime._context import LangGraphRuntimeContextBuilder
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._runtime import LangGraphRuntime
from ._utils._graph import LangGraphConfig

load_dotenv()


def langgraph_run_middleware(
    entrypoint: Optional[str], input: Optional[str], resume: bool, **kwargs
) -> MiddlewareResult:
    """Middleware to handle langgraph execution"""
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        # Add default env variables
        env["UIPATH_REQUESTING_PRODUCT"] = "uipath-python-sdk"
        env["UIPATH_REQUESTING_FEATURE"] = "langgraph-agent"

        context = (
            LangGraphRuntimeContextBuilder()
            .with_defaults(**kwargs)
            .with_langgraph_config(config)
            .with_entrypoint(entrypoint)
            .with_input(input_data=input, input_file=kwargs.get("input_file"))
            .with_resume(resume)
        ).build()

        async def execute():
            async with LangGraphRuntime.from_context(context) as runtime:
                if context.resume is False and context.job_id is None:
                    # Delete the previous graph state file at debug time
                    if os.path.exists(runtime.state_file_path):
                        os.remove(runtime.state_file_path)
                await runtime.execute()

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
