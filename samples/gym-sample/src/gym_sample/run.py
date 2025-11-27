"""Agent execution and evaluation runner."""

import asyncio
import argparse
import json
from typing import Dict, List, Tuple

from dotenv import find_dotenv, load_dotenv
from langgraph.graph import StateGraph
from uipath.eval.evaluators.base_evaluator import EvaluationResult
from uipath.eval.evaluators import LegacyBaseEvaluator
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.models import AgentExecution
from gym_sample.graph import agents_with_datapoints, get_all_evaluators
from gym_sample.trace_utils import setup_tracer
from gym_sample.uipath_gym_types import Datapoint


async def run_agents_with_tracing(graphs: List[Tuple[StateGraph, Datapoint]], verbose: bool = False, max_datapoints: int = 0) -> List[AgentExecution]:
    """Run the agent with OpenTelemetry trace collection across all datapoints.

    Note: This evaluation mode uses the unified graph approach where each graph accepts
    input at runtime via ainvoke(input_data). This allows both batch evaluation and
    CLI mode (uipath run calculator) to use the same graph building mechanism.

    Args:
        graphs: List of (StateGraph, Datapoint) tuples to run
        verbose: Whether to print verbose output

    Returns:
        List of AgentExecution: Contains agent input, output, and collected traces for each datapoint.
    """
    if verbose:
        print("Starting agent execution with trace collection...")

    results = []

    # Set up tracing once for all runs
    exporter, _ = setup_tracer()

    # Run the agent through all graphs/datapoints
    if not graphs:
        raise ValueError(f"No graphs found")

    if max_datapoints > 0:
        graphs = graphs[:max_datapoints]

    for i, (graph, datapoint) in enumerate(graphs):
        if verbose:
            print(f"\nRunning datapoint {i+1}/{len(graphs)}: {datapoint.input}")

        # Clear previous spans before each run
        exporter.clear_exported_spans()

        compiled_graph = graph.compile()
        # Unified graphs accept input at runtime - pass the datapoint input directly
        result = await compiled_graph.ainvoke(datapoint.input)

        # Extract output - with output_schema, result is the typed output directly
        if isinstance(result, dict):
            # Try to get the output from the result (handle both old and new format)
            agent_output = result if 'answer' in result or 'result' not in result else result.get('result', {})
        elif hasattr(result, 'model_dump'):
            # Result is a Pydantic model (output_schema) - convert to dict
            agent_output = result.model_dump()
        else:
            agent_output = {}

        agent_trace = exporter.get_exported_spans().copy()  # Copy to avoid reference issues

        if verbose:
            print(f"Agent completed. Result: {agent_output}")
            print(f"Collected {len(agent_trace)} trace spans")

        results.append(AgentExecution(
            agent_input=datapoint.input,
            agent_output=agent_output,
            agent_trace=agent_trace,
            simulation_instructions=datapoint.simulation_instructions,
        ))

    return results


async def run_evaluation(agent_name: str, include_llm_judge: bool = False, verbose: bool = False, max_datapoints: int = 0) -> Dict[str, Dict[str, EvaluationResult]]:
    """Run the complete agent evaluation pipeline across all datapoints."""
    print(f"Running evaluation for agent: {agent_name}")

    evaluators_generator = get_all_evaluators()[agent_name]
    evaluators = evaluators_generator(include_llm_judge)

    async with agents_with_datapoints(agent_name) as graphs:
        agent_executions = await run_agents_with_tracing(graphs, verbose, max_datapoints)
        datapoints = [datapoint for _, datapoint in graphs]

    if verbose:
        print(f"\nAgent executed successfully across {len(agent_executions)} datapoints!")
        print("\nRunning evaluations...")
    else:
        print("\nRunning evaluations...")

    all_results: Dict[str, Dict[str, EvaluationResult]] = {}

    # Get datapoints to access their evaluation criteria
    # Run evaluations for each datapoint execution
    for i, agent_execution in enumerate(agent_executions):
        datapoint = datapoints[i]
        datapoint_key = datapoint.name
        if verbose:
            print(f"\n--- Evaluating Datapoint {i+1}/{len(agent_executions)} ---")
            print(f"Input: {agent_execution.agent_input}")
            print(f"Output: {agent_execution.agent_output}")
            print(f"Collected {len(agent_execution.agent_trace)} trace spans")

        datapoint_results: Dict[str, EvaluationResult] = {}

        # Run each evaluator for this datapoint
        for evaluator in evaluators:
            if evaluator.name not in datapoint.evaluation_criteria:
                if verbose:
                    print(f"  {evaluator.name} not found in datapoint evaluation criteria. Skipping.")
                continue

            # Use the datapoint's evaluation criteria - key by evaluator.name
            evaluator_criteria = datapoint.evaluation_criteria[evaluator.name]
            if verbose:
                print(f"  Evaluating {evaluator.name} with criteria: {evaluator_criteria}")

            # Use the appropriate evaluate method based on evaluator type
            if isinstance(evaluator, LegacyBaseEvaluator):
                # Legacy evaluators use evaluate method
                result = await evaluator.evaluate(
                    agent_execution=agent_execution,
                    evaluation_criteria=evaluator_criteria
                )
            elif isinstance(evaluator, BaseEvaluator):
                # New evaluators use validate_and_evaluate_criteria
                result = await evaluator.validate_and_evaluate_criteria(
                    agent_execution=agent_execution,
                    evaluation_criteria=evaluator_criteria
                )
            else:
                raise ValueError(f"Evaluator {evaluator.__class__.__name__} is not a valid evaluator")

            if verbose:
                print(f"  Result: {result}")
            datapoint_results[evaluator.name] = result

        all_results[datapoint_key] = datapoint_results

    return all_results


async def async_main() -> None:
    """Main entry point for agent evaluation (async version)."""
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--agent_name", default="calculator", help="Name of the agent to run")
    parser.add_argument("--include_llm_judge", action="store_true", help="Include LLM judge evaluators")
    parser.add_argument("--max_datapoints", type=int, default=0, help="Maximum number of datapoints to run")
    args = parser.parse_args()

    load_dotenv(find_dotenv())

    results = await run_evaluation(agent_name=args.agent_name, include_llm_judge=args.include_llm_judge, verbose=args.verbose, max_datapoints=args.max_datapoints)

    # Print results for all datapoints
    summary = {}
    for datapoint_key, datapoint_results in results.items():
        summary[datapoint_key] = {evaluator_id: (result.score, str(result.details)) for evaluator_id, result in datapoint_results.items()}

    print(json.dumps(summary, indent=4))


def main() -> None:
    """Synchronous entry point for CLI script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
