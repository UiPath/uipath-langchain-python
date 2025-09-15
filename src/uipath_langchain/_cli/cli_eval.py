import asyncio
from os import environ as env
from typing import List, Optional

from uipath._cli._evals._runtime import UiPathEvalContext, UiPathEvalRuntime
from uipath._cli._runtime._contracts import (
    UiPathRuntimeFactory,
)
from uipath._cli._utils._eval_set import EvalHelpers
from uipath._cli.middlewares import MiddlewareResult
from uipath.eval._helpers import auto_discover_entrypoint

from uipath_langchain._cli._runtime._context import LangGraphRuntimeContext
from uipath_langchain._cli._runtime._runtime import LangGraphRuntime
from uipath_langchain._cli._utils._graph import LangGraphConfig


def langgraph_eval_middleware(
    entrypoint: Optional[str], eval_set: Optional[str], eval_ids: List[str], **kwargs
) -> MiddlewareResult:
    # Add default env variables
    env["UIPATH_REQUESTING_PRODUCT"] = "uipath-python-sdk"
    env["UIPATH_REQUESTING_FEATURE"] = "langgraph-agent"

    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    eval_context = UiPathEvalContext.with_defaults(**kwargs)
    eval_context.eval_set = eval_set or EvalHelpers.auto_discover_eval_set()
    eval_context.eval_ids = eval_ids

    try:
        runtime_entrypoint = entrypoint or auto_discover_entrypoint()

        def generate_runtime_context(
            context_entrypoint: str, langgraph_config: LangGraphConfig, **context_kwargs
        ) -> LangGraphRuntimeContext:
            context = LangGraphRuntimeContext.with_defaults(**context_kwargs)
            context.langgraph_config = langgraph_config
            context.entrypoint = context_entrypoint
            return context

        runtime_factory = UiPathRuntimeFactory(
            LangGraphRuntime,
            LangGraphRuntimeContext,
            context_generator=lambda **context_kwargs: generate_runtime_context(
                context_entrypoint=runtime_entrypoint,
                langgraph_config=config,
                **context_kwargs,
            ),
        )

        async def execute():
            async with UiPathEvalRuntime.from_eval_context(
                factory=runtime_factory, context=eval_context
            ) as eval_runtime:
                await eval_runtime.execute()

        asyncio.run(execute())
        return MiddlewareResult(should_continue=False)

    except Exception as e:
        return MiddlewareResult(
            should_continue=False, error_message=f"Error running evaluation: {str(e)}"
        )
