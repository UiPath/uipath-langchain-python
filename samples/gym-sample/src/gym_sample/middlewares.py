import asyncio

from uipath._cli._evals._console_progress_reporter import ConsoleProgressReporter
from uipath._cli._evals._runtime import UiPathEvalContext
from uipath._cli._utils._eval_set import EvalHelpers
from uipath._cli._utils._studio_project import StudioClient
from uipath._cli.middlewares import MiddlewareResult, Middlewares, console
from uipath._events._event_bus import EventBus
from uipath._utils._bindings import ResourceOverwritesContext
from uipath.core import UiPathTraceManager
from uipath.eval._helpers import auto_discover_entrypoint
from uipath.platform.common import UiPathConfig
from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
    UiPathRuntimeResult,
)

from gym_sample.eval_runtime import GymEvalRuntime


async def evaluate(
    runtime_factory: UiPathRuntimeFactoryProtocol,
    trace_manager: UiPathTraceManager,
    eval_context: UiPathEvalContext,
    event_bus: EventBus,
) -> UiPathRuntimeResult:
    async with GymEvalRuntime(
        factory=runtime_factory,
        context=eval_context,
        trace_manager=trace_manager,
        event_bus=event_bus,
    ) as eval_runtime:
        results = await eval_runtime.execute()
        await event_bus.wait_for_all(timeout=10)
        return results


def langgraph_eval_middleware(
    entrypoint: str | None,
    eval_set: str | None,
    eval_ids: list[str],
    eval_set_run_id: str | None = None,
    no_report: bool = False,
    workers: int = 1,
    output_file: str | None = None,
    register_progress_reporter: bool = False,
    enable_mocker_cache: bool = False,  # TODO: this doesn't exist!
    report_coverage: bool = False,  # TODO: this doesn't exist!
):
    eval_context = UiPathEvalContext()

    eval_context.entrypoint = entrypoint or auto_discover_entrypoint()
    eval_context.no_report = no_report
    eval_context.workers = workers
    eval_context.eval_set_run_id = eval_set_run_id
    eval_context.enable_mocker_cache = enable_mocker_cache

    # Load eval set to resolve the path
    # TODO: get from entrypoint?

    eval_set_path = eval_set or EvalHelpers.auto_discover_eval_set()
    _, resolved_eval_set_path = EvalHelpers.load_eval_set(eval_set_path, eval_ids)

    eval_context.eval_set = resolved_eval_set_path
    eval_context.eval_ids = eval_ids
    eval_context.report_coverage = report_coverage

    try:

        async def execute_eval():
            event_bus = EventBus()

            console_reporter = ConsoleProgressReporter()
            await console_reporter.subscribe_to_eval_runtime_events(event_bus)

            trace_manager = UiPathTraceManager()

            with UiPathRuntimeContext.with_defaults(
                output_file=output_file,
                trace_manager=trace_manager,
                command="eval",
            ) as ctx:
                project_id = UiPathConfig.project_id
                runtime_factory = UiPathRuntimeFactoryRegistry.get(context=ctx)

                try:
                    if project_id:
                        studio_client = StudioClient(project_id)

                        async with ResourceOverwritesContext(
                            lambda: studio_client.get_resource_overwrites()
                        ):
                            ctx.result = await evaluate(
                                runtime_factory,
                                trace_manager,
                                eval_context,
                                event_bus,
                            )
                    else:
                        # Fall back to execution without overwrites
                        ctx.result = await evaluate(
                            runtime_factory, trace_manager, eval_context, event_bus
                        )
                finally:
                    if runtime_factory:
                        await runtime_factory.dispose()

        asyncio.run(execute_eval())
        return MiddlewareResult(should_continue=False)  # TODO: messages
    except Exception as e:
        console.error(
            f"Error occurred: {e or 'Execution failed'}", include_traceback=True
        )
        return MiddlewareResult(should_continue=True)  # TODO: messages


def register_middleware():
    """This function will be called by the entry point system when uipath_langchain is installed"""
    Middlewares.register("eval", langgraph_eval_middleware)
