import asyncio
from os import environ as env
from typing import List, Optional

from uipath._cli._evals._runtime import UiPathEvalContext, UiPathEvalRuntime
from uipath._cli._runtime._contracts import (
    UiPathRuntimeContext,
    UiPathRuntimeContextBuilder,
    UiPathRuntimeFactory,
)
from uipath._cli.middlewares import MiddlewareResult

from uipath_langchain._cli._runtime._context import LangGraphRuntimeContextBuilder
from uipath_langchain._cli._runtime._runtime import LangGraphRuntime
from uipath_langchain._cli._utils._graph import LangGraphConfig


def langgraph_eval_middleware(
    entrypoint: Optional[str], eval_set: Optional[str], eval_ids: List[str], **kwargs
) -> MiddlewareResult:
    def generate_eval_context(
        runtime_context: UiPathRuntimeContext,
    ) -> UiPathEvalContext[UiPathRuntimeContext]:
        base_context = UiPathRuntimeContextBuilder().with_defaults().build()
        return UiPathEvalContext(
            runtime_context=runtime_context,
            no_report=kwargs.get("no_report", False),
            workers=kwargs.get("workers", 8),
            eval_set=eval_set,
            eval_ids=eval_ids,
            **kwargs,
            **base_context.model_dump(),
        )

    try:
        runtime_factory = UiPathRuntimeFactory(LangGraphRuntime, UiPathRuntimeContext)
        # Add default env variables
        env["UIPATH_REQUESTING_PRODUCT"] = "uipath-python-sdk"
        env["UIPATH_REQUESTING_FEATURE"] = "langgraph-agent"

        context = (
            LangGraphRuntimeContextBuilder()
            .with_defaults(**kwargs)
            .with_langgraph_config(LangGraphConfig())
            .with_entrypoint(entrypoint)
            .mark_eval_run()
        ).build()

        async def execute():
            async with UiPathEvalRuntime.from__eval_context(
                factory=runtime_factory, context=generate_eval_context(context)
            ) as eval_runtime:
                await eval_runtime.execute()

        asyncio.run(execute())
        return MiddlewareResult(should_continue=False)

    except Exception as e:
        return MiddlewareResult(
            should_continue=False, error_message=f"Error running evaluation: {str(e)}"
        )
