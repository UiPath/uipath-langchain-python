import uuid
from collections import defaultdict

import coverage
from uipath._cli._evals._models._output import UiPathEvalOutput
from uipath._cli._evals._runtime import (
    ExecutionLogsExporter,
    ExecutionSpanExporter,
    UiPathEvalContext,
    UiPathEvalRuntime,
    execute_parallel,
)
from uipath._cli._evals._span_collection import ExecutionSpanCollector
from uipath._events._event_bus import EventBus
from uipath._events._events import EvalSetRunUpdatedEvent, EvaluationEvents
from uipath.core import UiPathTraceManager
from uipath.runtime import (
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeResult,
    UiPathRuntimeSchema,
    UiPathRuntimeStatus,
)


class GymEvalRuntimeWithSingleArgs(UiPathEvalRuntime):
    """Same as base class, with different initialization."""

    def __init__(
        self,
        context: UiPathEvalContext,
        factory: UiPathRuntimeFactoryProtocol,
        trace_manager: UiPathTraceManager,
        span_exporter: ExecutionSpanExporter,
        span_collector: ExecutionSpanCollector,
        event_bus: EventBus,
    ):
        self.context: UiPathEvalContext = context
        self.factory: UiPathRuntimeFactoryProtocol = factory
        self.event_bus: EventBus = event_bus
        self.trace_manager: UiPathTraceManager = trace_manager
        self.span_exporter = span_exporter
        self.span_collector = span_collector

        self.logs_exporter: ExecutionLogsExporter = ExecutionLogsExporter()
        self.execution_id = str(uuid.uuid4())
        self.schema: UiPathRuntimeSchema | None = None
        self.coverage = coverage.Coverage(branch=True)


class GymEvalRuntime(UiPathEvalRuntime):
    """Specialized runtime for evaluation runs, with access to the factory."""

    async def execute(self) -> UiPathRuntimeResult:
        with self._mocker_cache():
            agent_evaluation_runtime = GymEvalRuntimeWithSingleArgs(
                context=self.context,
                factory=self.factory,
                trace_manager=self.trace_manager,
                span_exporter=self.span_exporter,
                span_collector=self.span_collector,
                event_bus=self.event_bus,
            )
            (
                evaluation_set,
                evaluators,
                evaluation_iterable,
            ) = await agent_evaluation_runtime.initiate_evaluation()
            workers = self.context.workers or 1
            assert workers >= 1
            eval_run_result_list = await execute_parallel(evaluation_iterable, workers)
            results = UiPathEvalOutput(
                evaluation_set_name=evaluation_set.name,
                evaluation_set_results=eval_run_result_list,
            )

            # Computing evaluator averages
            evaluator_averages: dict[str, float] = defaultdict(float)
            evaluator_count: dict[str, int] = defaultdict(int)

            # Check if any eval runs failed
            any_failed = False
            for eval_run_result in results.evaluation_set_results:
                # Check if the agent execution had an error
                if (
                    eval_run_result.agent_execution_output
                    and eval_run_result.agent_execution_output.result.error
                ):
                    any_failed = True

                for result_dto in eval_run_result.evaluation_run_results:
                    evaluator_averages[result_dto.evaluator_id] += (
                        result_dto.result.score
                    )
                    evaluator_count[result_dto.evaluator_id] += 1

            for eval_id in evaluator_averages:
                evaluator_averages[eval_id] = (
                    evaluator_averages[eval_id] / evaluator_count[eval_id]
                )
            await self.event_bus.publish(
                EvaluationEvents.UPDATE_EVAL_SET_RUN,
                EvalSetRunUpdatedEvent(
                    execution_id=self.execution_id,
                    evaluator_scores=evaluator_averages,
                    success=not any_failed,
                ),
                wait_for_completion=False,
            )

            result = UiPathRuntimeResult(
                output={**results.model_dump(by_alias=True)},
                status=UiPathRuntimeStatus.SUCCESSFUL,
            )
            return result
