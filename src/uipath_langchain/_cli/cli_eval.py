import asyncio
from typing import List, Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._evals._console_progress_reporter import ConsoleProgressReporter
from uipath._cli._evals._evaluate import evaluate
from uipath._cli._evals._progress_reporter import StudioWebProgressReporter
from uipath._cli._evals._runtime import UiPathEvalContext
from uipath._cli._runtime._contracts import (
    UiPathRuntimeContext,
    UiPathRuntimeFactory,
)
from uipath._cli._utils._eval_set import EvalHelpers
from uipath._cli.middlewares import MiddlewareResult
from uipath._events._event_bus import EventBus
from uipath.eval._helpers import auto_discover_entrypoint
from uipath.tracing import LlmOpsHttpExporter

from uipath_langchain._cli._runtime._memory import get_memory
from uipath_langchain._cli._runtime._runtime import LangGraphScriptRuntime
from uipath_langchain._cli._utils._graph import LangGraphConfig
from uipath_langchain._tracing import (
    _instrument_traceable_attributes,
)


def langgraph_eval_middleware(
    entrypoint: Optional[str], eval_set: Optional[str], eval_ids: List[str], **kwargs
) -> MiddlewareResult:
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        _instrument_traceable_attributes()

        async def execute():
            event_bus = EventBus()

            if kwargs.get("register_progress_reporter", False):
                progress_reporter = StudioWebProgressReporter(
                    spans_exporter=LlmOpsHttpExporter(extra_process_spans=True)
                )
                await progress_reporter.subscribe_to_eval_runtime_events(event_bus)

            console_reporter = ConsoleProgressReporter()
            await console_reporter.subscribe_to_eval_runtime_events(event_bus)

            def generate_runtime_context(
                context_entrypoint: str, **context_kwargs
            ) -> UiPathRuntimeContext:
                context = UiPathRuntimeContext.with_defaults(**context_kwargs)
                context.entrypoint = context_entrypoint
                return context

            runtime_entrypoint = entrypoint or auto_discover_entrypoint()

            eval_context = UiPathEvalContext.with_defaults(
                entrypoint=runtime_entrypoint, **kwargs
            )
            eval_context.eval_set = eval_set or EvalHelpers.auto_discover_eval_set()
            eval_context.eval_ids = eval_ids

            async with get_memory(eval_context) as memory:
                runtime_factory = UiPathRuntimeFactory(
                    LangGraphScriptRuntime,
                    UiPathRuntimeContext,
                    context_generator=lambda **context_kwargs: generate_runtime_context(
                        context_entrypoint=runtime_entrypoint,
                        **context_kwargs,
                    ),
                    runtime_generator=lambda ctx: LangGraphScriptRuntime(
                        ctx, memory, ctx.entrypoint
                    ),
                )

                if eval_context.job_id:
                    runtime_factory.add_span_exporter(
                        LlmOpsHttpExporter(extra_process_spans=True)
                    )

                runtime_factory.add_instrumentor(
                    LangChainInstrumentor, get_current_span
                )

                await evaluate(runtime_factory, eval_context, event_bus)

        asyncio.run(execute())

        return MiddlewareResult(should_continue=False)

    except Exception as e:
        return MiddlewareResult(
            should_continue=False, error_message=f"Error running evaluation: {str(e)}"
        )
