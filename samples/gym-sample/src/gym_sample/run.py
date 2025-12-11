"""Agent execution and evaluation runner."""

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Generator, Sequence
from pathlib import Path
from typing import Any, Callable

from aiostream import pipe, stream
from dotenv import find_dotenv, load_dotenv
from langgraph.graph import StateGraph
from opentelemetry.trace import TracerProvider
from pydantic import BaseModel
from tqdm.std import tqdm  # type:ignore[import-untyped]
from uipath._cli._evals._models._evaluation_set import EvaluationItem
from uipath._cli._evals._models._output import EvaluationRunResult
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import EvaluationResult
from uipath.eval.models import AgentExecution

from gym_sample.graph import agents_with_datapoints_generator, get_all_evaluators
from gym_sample.trace_utils import InMemorySpanExporter, setup_tracer


def sliced_generator[T](
    generator: Generator[T, None, None], max_items: int
) -> Generator[T, None, None]:
    count = 0
    for item in generator:
        if count >= max_items:
            break
        yield item
        count += 1


class AgentDatapointExecution(BaseModel):
    datapoint: EvaluationItem
    agent_execution: AgentExecution


async def gather_limited[T, R](
    limit: int,
    items: Generator[T, None, None] | Sequence[T],
    func: Callable[..., Awaitable[R]],
    *,
    total: int | None = None,  # length hint for tqdm
    show_progress: bool = True,
) -> list[R]:
    async def _wrapper(item: Any, *args: Any) -> R:
        # Your current convention: each item is a tuple of args
        return await func(*item)

    # If it's a sequence and total wasn't given, infer it
    if isinstance(items, Sequence) and total is None:
        total = len(items)

    # For a generator without total, tqdm can't show nice progress
    # (we keep streaming behavior, just no bar)
    s = stream.iterate(items)
    mapped = s | pipe.map(_wrapper, task_limit=limit)

    results: list[R] = []

    if show_progress and total is not None:
        pbar = tqdm(total=total)
        try:
            async for x in mapped:
                results.append(x)
                pbar.update(1)
        finally:
            pbar.close()
    else:
        # Fallback: no progress bar
        results = [x async for x in mapped]

    return results


async def run_agent_with_tracing(
    graph: StateGraph[Any, Any],
    datapoint: EvaluationItem,
    span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
) -> AgentDatapointExecution:
    """Run the agent with OpenTelemetry trace collection."""
    compiled_graph = graph.compile()
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span(name="agent_execution") as root_span:
        trace_id = root_span.get_span_context().trace_id
        result = await compiled_graph.ainvoke(datapoint.inputs)

    spans = span_exporter.get_exported_spans()
    filtered_spans = [
        span
        for span in spans
        if (ctx := span.get_span_context()) is not None and ctx.trace_id == trace_id
    ]
    agent_execution = AgentExecution(
        agent_input=datapoint.inputs,
        agent_output=result,
        agent_trace=filtered_spans,
        simulation_instructions=datapoint.input_mocking_strategy.prompt
        if datapoint.input_mocking_strategy
        else "",
    )
    return AgentDatapointExecution(
        datapoint=datapoint,
        agent_execution=agent_execution,
    )


async def run_agents_with_tracing_parallel(
    graphs: Generator[tuple[StateGraph[Any, Any], EvaluationItem], None, None],
    num_datapoints: int,
    concurrency: int = 20,
    verbose: bool = False,
) -> list[AgentDatapointExecution]:
    """Run the agent with OpenTelemetry trace collection across all datapoints in parallel."""
    if verbose:
        print("Starting agent execution with trace collection...")

    # Set up tracing once for all runs
    exporter, tracer_provider = setup_tracer()

    # Run the agent through all graphs/datapoints
    if not graphs:
        raise ValueError("No graphs found")

    def run_agent_with_tracing_fn(
        graph: StateGraph[Any, Any], datapoint: EvaluationItem
    ) -> Awaitable[AgentDatapointExecution]:
        return run_agent_with_tracing(graph, datapoint, exporter, tracer_provider)

    results = await gather_limited(
        concurrency,
        graphs,
        run_agent_with_tracing_fn,
        total=num_datapoints,
        show_progress=True,
    )

    return results


async def run_evaluation_parallel(
    agent_name: str,
    agent_datapoint_executions: list[AgentDatapointExecution],
    concurrency: int = 20,
    include_llm_judge: bool = False,
    verbose: bool = False,
) -> dict[str, dict[str, EvaluationResult]]:
    evaluators_generator = get_all_evaluators()[agent_name]
    evaluators = evaluators_generator(include_llm_judge)
    args: list[tuple[BaseEvaluator[Any, Any, Any], AgentDatapointExecution]] = []
    for agent_datapoint_execution in agent_datapoint_executions:
        evaluation_criterias = agent_datapoint_execution.datapoint.evaluation_criterias
        if extra_evaluators := set[str](evaluation_criterias.keys()).difference(
            evaluators.keys()
        ):
            raise ValueError(
                f"Evaluators {extra_evaluators} do not exist in the evaluators dictionary"
            )
        for evaluator_id in evaluation_criterias:
            args.append((evaluators[evaluator_id], agent_datapoint_execution))

    async def run_evaluator_fn(
        evaluator: BaseEvaluator[Any, Any, Any], agent_datapoint_execution: AgentDatapointExecution
    ) -> tuple[str, str, EvaluationResult]:
        datapoint = agent_datapoint_execution.datapoint
        agent_execution = agent_datapoint_execution.agent_execution

        evaluator_criteria = datapoint.evaluation_criterias[evaluator.id]
        if verbose:
            print(f"  Evaluating {evaluator.id} with criteria: {evaluator_criteria}")

        result = await evaluator.validate_and_evaluate_criteria(
            agent_execution=agent_execution, evaluation_criteria=evaluator_criteria
        )

        return (evaluator.id, datapoint.name, result)

    results = await gather_limited(
        concurrency, args, run_evaluator_fn, total=len(args), show_progress=True
    )
    results_dict: dict[str, dict[str, EvaluationResult]] = defaultdict(dict)
    for evaluator_name, datapoint_name, result in results:
        results_dict[evaluator_name][datapoint_name] = result
    return dict(results_dict)


def save_evaluation_results(
    agent_name: str,
    results: dict[str, dict[str, EvaluationResult]],
    agent_version: str = "1.0.0",
    loop_name: str = "default_loop",
    loop_version: str = "1.0.0",
    output_dir: Path | None = None,
) -> Path:
    """Save evaluation results to disk using the AgentEvaluationResults BaseModel.

    Args:
        agent_name: Name of the agent being evaluated
        results: Nested dict mapping evaluator_id -> datapoint_name -> EvaluationResult
        agent_version: Version of the agent (default: "1.0.0")
        loop_name: Name of the loop (default: "default_loop")
        loop_version: Version of the loop (default: "1.0.0")
        output_dir: Optional directory to save results. Defaults to './results'

    Returns:
        Path to the saved results file
    """
    if output_dir is None:
        output_dir = Path("./results")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert the nested dict structure to a flat list of RawResultSchema objects
    evaluation_entries = []
    for evaluator_id, datapoint_results in results.items():
        for datapoint_name, result in datapoint_results.items():
            # Create experiment unique identifier
            experiment_id = f"{agent_name}_{loop_name}_{evaluator_id}"

            entry = EvaluationRunResult(
                experiment_unique_identifier=experiment_id,
                agent=agent_name,
                agent_version=agent_version,
                loop_name=loop_name,
                loop_version=loop_version,
                testcase=datapoint_name,
                score=result.score,
                metric=evaluator_id,
                metric_tags=[],  # UIPath SDK doesn't provide metric tags
                agent_metric_weights={},  # UIPath SDK doesn't provide metric weights
                split="TODO",  # Default to test split
            )
            evaluation_entries.append(entry)

    output_file = output_dir / "evaluation_results.json"

    # Convert each entry to dict and save as a plain JSON array
    results_list = [entry.model_dump() for entry in evaluation_entries]

    with open(output_file, "w") as f:
        json.dump(results_list, f, indent=2)

    print(f"\n✅ Evaluation results saved to: {output_file}")
    return output_file


async def run_evaluation(
    agent_name: str,
    concurrency: int = 20,
    include_llm_judge: bool = False,
    verbose: bool = False,
    max_datapoints: int = 0,
) -> dict[str, dict[str, EvaluationResult]]:
    """Run the complete agent evaluation pipeline across all datapoints."""
    print(f"Running evaluation for agent: {agent_name}")

    graphs_generator, num_datapoints = agents_with_datapoints_generator(
        agent_name, max_datapoints
    )
    agent_datapoint_executions = await run_agents_with_tracing_parallel(
        graphs_generator, num_datapoints, concurrency, verbose
    )

    if verbose:
        print(
            f"\nAgent executed successfully across {len(agent_datapoint_executions)} datapoints!"
        )
        print("\nRunning evaluations...")
    else:
        print("\nRunning evaluations...")
    all_results = await run_evaluation_parallel(
        agent_name, agent_datapoint_executions, concurrency, include_llm_judge, verbose
    )

    return all_results


async def async_main() -> None:
    """Main entry point for agent evaluation (async version)."""
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--agent_name", default="calculator", help="Name of the agent to run"
    )
    parser.add_argument(
        "--include_llm_judge", action="store_true", help="Include LLM judge evaluators"
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=0,
        help="Maximum number of datapoints to run",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent tasks (default: 20)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results)",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv())

    results = await run_evaluation(
        agent_name=args.agent_name,
        concurrency=args.concurrency,
        include_llm_judge=args.include_llm_judge,
        verbose=args.verbose,
        max_datapoints=args.max_datapoints,
    )

    # Save results to disk instead of printing to CLI
    output_file = save_evaluation_results(
        agent_name=args.agent_name,
        results=results,
        output_dir=Path(args.output_dir),
    )

    # Print a brief summary to CLI
    total_evaluations = sum(len(dp_results) for dp_results in results.values())
    print("\n📊 Summary:")
    print(f"   Total evaluations: {total_evaluations}")
    print(f"   Evaluators: {len(results)}")
    print(f"   Results file: {output_file}")


def main() -> None:
    """Synchronous entry point for CLI script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
