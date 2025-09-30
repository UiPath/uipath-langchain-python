"""Agent execution and evaluation runner."""

import asyncio
import argparse
from datetime import datetime
import json
from typing import Dict, List, Tuple

from dotenv import find_dotenv, load_dotenv
from langgraph.graph import StateGraph
from uipath.eval.evaluators.base_evaluator import EvaluatorCategory, EvaluatorType
from uipath.eval.models import EvaluationResult

from gym_sample.evaluators import LLMJudgeEvaluator, LLMJudgeSimulationTrajectoryEvaluator, LLMJudgeStrictJSONSimilarityEvaluator, LLMJudgeTrajectoryEvaluator, ToolCallArgumentsEvaluator, ToolCallCountEvaluator, ToolCallOrderEvaluator, ToolCallOutputEvaluator
from uipath.eval.evaluators import (
    BaseEvaluator,
    ExactMatchEvaluator,
)
from gym_sample.graph import agents_with_datapoints
from gym_sample.trace_utils import setup_tracer
from gym_sample.evaluators_helpers import AgentExecution
from gym_sample.uipath_gym_types import Datapoint


async def run_agents_with_tracing(graphs: List[Tuple[StateGraph, Datapoint]], verbose: bool = False) -> List[AgentExecution]:
    """Run the agent with OpenTelemetry trace collection across all datapoints.

    Note: This evaluation mode bypasses the CLI entry point and directly builds graphs
    for each datapoint. This allows batch evaluation while the CLI mode (uipath run calculator)
    accepts properly typed GraphInput models for single executions.

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

    for i, (graph, datapoint) in enumerate(graphs):
        if verbose:
            print(f"\nRunning datapoint {i+1}/{len(graphs)}: {datapoint.input}")

        # Clear previous spans before each run
        exporter.clear_exported_spans()

        compiled_graph = graph.compile()
        # Evaluation graphs have input pre-bound at build time, so pass empty dict
        result = await compiled_graph.ainvoke({})

        # Extract output and get spans only from this run
        agent_output = result.get('result', {}) if isinstance(result, dict) else {}
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


def create_evaluators(include_llm_judge: bool = False) -> List[BaseEvaluator]:
    """Create evaluators and their evaluation criteria.

    Returns:
        List of evaluators.
    """
    now = datetime.now().isoformat()

    exact_match_evaluator = ExactMatchEvaluator(
        id="exact_match",
        name="Exact Match",
        created_at=now,
        updated_at=now,
        description="Evaluates if the actual output exactly matches the expected output",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Equals,
    )

    tool_call_order_evaluator = ToolCallOrderEvaluator(
        id="tool_call_order",
        name="Tool Call Order",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct order",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
    )

    tool_call_count_evaluator = ToolCallCountEvaluator(
        id="tool_call_count",
        name="Tool Call Count",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct count",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
    )

    tool_call_arguments_evaluator = ToolCallArgumentsEvaluator(
        id="tool_call_arguments",
        name="Tool Call Arguments",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct arguments",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
        subset=False,
    )

    tool_call_output_evaluator = ToolCallOutputEvaluator(
        id="tool_call_output",
        name="Tool Call Output",
        created_at=now,
        updated_at=now,
        description="Evaluates if the tool calls are in the correct output",
        category=EvaluatorCategory.Deterministic,
        evaluator_type=EvaluatorType.Trajectory,
        strict=False,
    )

    llm_judge_evaluator = LLMJudgeEvaluator(
        id="llm_judge",
        name="LLM Judge",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.LlmAsAJudge,
        evaluator_type=EvaluatorType.Custom,
        model="gpt-4o-2024-11-20",
    )

    llm_judge_strict_json_similarity_evaluator = LLMJudgeStrictJSONSimilarityEvaluator(
        id="llm_judge_strict_json_similarity",
        name="LLM Judge Strict JSON Similarity",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.LlmAsAJudge,
        evaluator_type=EvaluatorType.Custom,
        model="gpt-4o-2024-11-20",
    )

    llm_judge_trajectory_evaluator = LLMJudgeTrajectoryEvaluator(
        id="llm_judge_trajectory",
        name="LLM Judge Trajectory",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.LlmAsAJudge,
        evaluator_type=EvaluatorType.Custom,
        model="gpt-4o-2024-11-20",
    )

    llm_judge_simulation_trajectory_evaluator = LLMJudgeSimulationTrajectoryEvaluator(
        id="llm_judge_simulation_trajectory",
        name="LLM Judge Simulation Trajectory",
        created_at=now,
        updated_at=now,
        description="Evaluates the output of the agent using an LLM",
        category=EvaluatorCategory.Trajectory,
        evaluator_type=EvaluatorType.Trajectory,
        model="gpt-4o-2024-11-20",
    )

    evaluators = [exact_match_evaluator, tool_call_order_evaluator, tool_call_count_evaluator, tool_call_arguments_evaluator, tool_call_output_evaluator]
    if include_llm_judge:
        evaluators.extend([llm_judge_evaluator, llm_judge_strict_json_similarity_evaluator, llm_judge_trajectory_evaluator, llm_judge_simulation_trajectory_evaluator])

    return evaluators


async def run_evaluation(agent_name: str, evaluators: List[BaseEvaluator], verbose: bool = False) -> Dict[str, Dict[str, EvaluationResult]]:
    """Run the complete agent evaluation pipeline across all datapoints."""
    print(f"Running evaluation for agent: {agent_name}")

    async with agents_with_datapoints(agent_name) as graphs:
        agent_executions = await run_agents_with_tracing(graphs, verbose)
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
        datapoint_key = f"datapoint_{i}"
        if verbose:
            print(f"\n--- Evaluating Datapoint {i+1}/{len(agent_executions)} ---")
            print(f"Input: {agent_execution.agent_input}")
            print(f"Output: {agent_execution.agent_output}")
            print(f"Collected {len(agent_execution.agent_trace)} trace spans")

        datapoint_results: Dict[str, EvaluationResult] = {}

        # Run each evaluator for this datapoint
        for evaluator in evaluators:
            # Use the datapoint's evaluation criteria
            evaluator_criteria = datapoint.evaluation_criteria.get(evaluator.id, {})
            if verbose:
                print(f"  Evaluating {evaluator.name} with criteria: {evaluator_criteria}")
            result = await evaluator.evaluate(
                agent_execution=agent_execution,
                evaluation_criteria=evaluator_criteria
            )
            if verbose:
                print(f"  Result: {result}")
            datapoint_results[evaluator.id] = result

        all_results[datapoint_key] = datapoint_results

    return all_results


async def main() -> None:
    """Main entry point for agent evaluation."""
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--agent_name", default="calculator", help="Name of the agent to run")
    parser.add_argument("--include_llm_judge", action="store_true", help="Include LLM judge evaluators")
    args = parser.parse_args()

    load_dotenv(find_dotenv())

    evaluators = create_evaluators(include_llm_judge=True)

    results = await run_evaluation(agent_name=args.agent_name, evaluators=evaluators, verbose=args.verbose)

    # Print results for all datapoints
    summary = {}
    for datapoint_key, datapoint_results in results.items():
        summary[datapoint_key] = {evaluator_id: (result.score, result.details) for evaluator_id, result in datapoint_results.items()}

    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
