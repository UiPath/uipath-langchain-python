import asyncio
import os
from typing import List, Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._evals._console_progress_reporter import ConsoleProgressReporter
from uipath._cli._evals._progress_reporter import StudioWebProgressReporter
from uipath._cli._evals._runtime import UiPathEvalContext, UiPathEvalRuntime
from uipath._cli._runtime._contracts import (
    UiPathRuntimeContext,
    UiPathRuntimeFactory,
)
from uipath._cli._utils._eval_set import EvalHelpers
from uipath._cli.middlewares import MiddlewareResult
from uipath._events._event_bus import EventBus
from uipath.eval._helpers import auto_discover_entrypoint

from uipath_langchain._cli._runtime._context import LangGraphRuntimeContext
from uipath_langchain._cli._runtime._runtime import LangGraphScriptRuntime
from uipath_langchain._cli._utils._graph import LangGraphConfig
from uipath_langchain._tracing import (
    LangChainExporter,
    _instrument_traceable_attributes,
)


def get_connection_string(context: UiPathRuntimeContext) -> str:
    if context.runtime_dir and context.state_file:
        os.makedirs(context.runtime_dir, exist_ok=True)
        return os.path.join(context.runtime_dir, context.state_file)
    return os.path.join("__uipath", "state.db")


def langgraph_eval_middleware(
    entrypoint: Optional[str], eval_set: Optional[str], eval_ids: List[str], **kwargs
) -> MiddlewareResult:
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:

        async def execute():
            _instrument_traceable_attributes()

            event_bus = EventBus()

            if kwargs.get("register_progress_reporter", False):
                progress_reporter = StudioWebProgressReporter(
                    spans_exporter=LangChainExporter()
                )
                await progress_reporter.subscribe_to_eval_runtime_events(event_bus)

            console_reporter = ConsoleProgressReporter()
            await console_reporter.subscribe_to_eval_runtime_events(event_bus)

            runtime_entrypoint = entrypoint or auto_discover_entrypoint()

            eval_context = UiPathEvalContext.with_defaults(
                entrypoint=runtime_entrypoint, **kwargs
            )
            eval_context.eval_set = eval_set or EvalHelpers.auto_discover_eval_set()
            eval_context.eval_ids = eval_ids

            def generate_runtime(
                ctx: LangGraphRuntimeContext,
            ) -> LangGraphScriptRuntime:
                return LangGraphScriptRuntime(ctx, ctx.entrypoint)

            def generate_runtime_context(
                context_entrypoint: str,
                context_memory: AsyncSqliteSaver,
                **context_kwargs,
            ) -> LangGraphRuntimeContext:
                context = LangGraphRuntimeContext.with_defaults(**context_kwargs)
                context.entrypoint = context_entrypoint
                context.memory = context_memory
                return context

            async with AsyncSqliteSaver.from_conn_string(
                get_connection_string(eval_context)
            ) as memory:
                runtime_factory = UiPathRuntimeFactory(
                    LangGraphScriptRuntime,
                    LangGraphRuntimeContext,
                    context_generator=lambda **context_kwargs: generate_runtime_context(
                        context_entrypoint=runtime_entrypoint,
                        context_memory=memory,
                        **context_kwargs,
                    ),
                    runtime_generator=generate_runtime,
                )

                if eval_context.job_id:
                    runtime_factory.add_span_exporter(LangChainExporter())

                runtime_factory.add_instrumentor(
                    LangChainInstrumentor, get_current_span
                )

                async with UiPathEvalRuntime.from_eval_context(
                    factory=runtime_factory, context=eval_context, event_bus=event_bus
                ) as eval_runtime:
                    await eval_runtime.execute()
                    await event_bus.wait_for_all()

        asyncio.run(execute())
        return MiddlewareResult(should_continue=False)

    except Exception as e:
        return MiddlewareResult(
            should_continue=False, error_message=f"Error running evaluation: {str(e)}"
        )
