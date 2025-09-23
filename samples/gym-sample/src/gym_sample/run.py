"""Agent execution and evaluation runner."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from uipath.eval.evaluators.base_evaluator import EvaluatorCategory, EvaluatorType

from gym_sample.evals import LLMJudgeEvaluator, LLMJudgeSimulationTrajectoryEvaluator, LLMJudgeStrictJSONSimilarityEvaluator, LLMJudgeTrajectoryEvaluator, ToolCallArgumentsEvaluator, ToolCallCountEvaluator, ToolCallOrderEvaluator
from uipath.eval.evaluators import (
    BaseEvaluator,
    ExactMatchEvaluator,
)
from gym_sample.graph import make_graph
from gym_sample.trace_utils import setup_tracer
from gym_sample.evals_helpers import AgentExecution


async def run_agent_with_tracing(agent_input: Dict[str, Any], simulation_instructions: str = "") -> AgentExecution:
    """Run the agent with OpenTelemetry trace collection.

    Returns:
        AgentExecution: Contains agent input, output, and collected traces.
    """
    # Set up tracing
    exporter, _ = setup_tracer()

    print("Starting agent execution with trace collection...")

    # Run the agent through the actual graph
    async with make_graph(agent_input) as graph:
        # Compile and run the graph
        compiled_graph = graph.compile()
        result = await compiled_graph.ainvoke(agent_input)

        # Extract output
        agent_output = result.get('result', {}) if isinstance(result, dict) else {}
        agent_trace = exporter.get_exported_spans()

        print(f"Agent completed. Result: {agent_output}")
        print(f"Collected {len(agent_trace)} trace spans")

        return AgentExecution(
            agent_input=agent_input,
            agent_output=agent_output,
            agent_trace=agent_trace,
            simulation_instructions=simulation_instructions,
        )


def create_evaluators() -> List[BaseEvaluator]:
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

    evaluators = [exact_match_evaluator, tool_call_order_evaluator, tool_call_count_evaluator, tool_call_arguments_evaluator, llm_judge_evaluator, llm_judge_strict_json_similarity_evaluator, llm_judge_trajectory_evaluator, llm_judge_simulation_trajectory_evaluator]

    return evaluators


async def run_evaluation(agent_input: Dict[str, Any], evaluation_criteria: Dict[str, Any], evaluators: List[BaseEvaluator], simulation_instructions: str = "") -> None:
    """Run the complete agent evaluation pipeline."""
    print("Running agent with real trace collection...")

    # Execute agent with tracing
    agent_execution = await run_agent_with_tracing(agent_input, simulation_instructions)

    print(f"\nAgent executed successfully!")
    print(f"Input: {agent_execution.agent_input}")
    print(f"Output: {agent_execution.agent_output}")
    print(f"Collected {len(agent_execution.agent_trace)} trace spans")

    print("\nRunning evaluations...")

    # Run each evaluator
    for evaluator in evaluators:
        print(f"\nEvaluating {evaluator.name} with criteria: {evaluation_criteria[evaluator.id]}")
        result = await evaluator.evaluate(
            agent_execution=agent_execution,
            evaluation_criteria=evaluation_criteria[evaluator.id]
        )
        print(f"Result: {result}")


async def main() -> None:
    """Main entry point for agent evaluation."""
    load_dotenv(find_dotenv())

    agent_input = {
        "expression": "15.0 + 7.0 * 3.0"
    }

    simulation_instructions = "Tool multiply should return 21.0 and tool add should return 36.0."

    evaluation_criteria = {
        "exact_match": {"answer": 36.0},
        "tool_call_order": ["multiply", "add"],
        "tool_call_count": {"multiply": "ge:1", "add": "ge:1"},
        "tool_call_arguments": [{"name": "multiply", "args": {"a": 7., "b":3.}}, {"name": "add", "args": {"a": 15., "b": 21.}}],
        "llm_judge": {"answer": 36.0},
        "llm_judge_strict_json_similarity": {"answer": 36.0},
        "llm_judge_trajectory": "The agent should have called the multiply tool with the arguments 7.0 and 3.0, and the add tool with the arguments 15.0 and 21.0.",
        "llm_judge_simulation_trajectory": "The agent should have called the multiply tool with the arguments 7.0 and 3.0, and the add tool with the arguments 15.0 and 21.0.",
    }

    evaluators = create_evaluators()

    await run_evaluation(agent_input, evaluation_criteria, evaluators, simulation_instructions)


if __name__ == "__main__":
    asyncio.run(main())
