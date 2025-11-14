"""Export AgentBaseClass evaluators and datapoints to UiPath eval format.

This script exports:
1. Evaluator specs in the new schema format (version 1.0) to evals/evaluators/
2. Eval sets that reference those evaluators to evals/eval-sets/
"""

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from gym_sample.uipath_gym_types import Datapoint
from uipath.eval.evaluators import BaseEvaluator

from gym_sample.graph import get_agents, get_all_evaluators


def evaluator_to_spec(evaluator: BaseEvaluator, agent_name: str) -> Dict[str, Any]:
    """Convert an evaluator instance to UiPath evaluator spec format.

    Args:
        evaluator: The evaluator instance
        agent_name: Name of the agent this evaluator belongs to

    Returns:
        Dict containing the evaluator spec in UiPath format (version 1.0)
    """
    # Get the evaluator ID from the get_evaluator_id class method
    evaluator_type_id = evaluator.get_evaluator_id()

    # Use agent-specific evaluator ID to avoid conflicts between agents
    # Each agent may need different config (e.g., targetOutputKey)
    # Use evaluator.id (not .name) to preserve numeric suffixes like _1, _2
    evaluator_id = f"{agent_name}_{evaluator.id}"

    # Use the evaluator_config (Pydantic model) and dump with aliases (camelCase)
    evaluator_config = evaluator.evaluator_config.model_dump(
        by_alias=True,
        exclude_none=True,
        exclude_unset=False  # Include defaults like targetOutputKey
    )

    spec = {
        "version": "1.0",
        "id": evaluator_id,
        "description": f"{evaluator.id} for {agent_name} agent",
        "evaluatorTypeId": evaluator_type_id,
        "evaluatorConfig": evaluator_config,
    }

    # Check if this is a custom evaluator (not in the standard SUPPORTED_EVALUATORS list)
    # For custom evaluators, we need to add the evaluatorSchema field
    BUILT_IN_EVALUATORS = {
        "ContainsEvaluator",
        "ExactMatchEvaluator",
        "JsonSimilarityEvaluator",
        "LLMJudgeOutputEvaluator",
        "LLMJudgeStrictJSONSimilarityOutputEvaluator",
        "ToolCallOrderEvaluator",
        "ToolCallCountEvaluator",
        "ToolCallArgsEvaluator",
        "ToolCallOutputEvaluator",
        "LLMJudgeTrajectoryEvaluator",
        "LLMJudgeTrajectorySimulationEvaluator",
    }

    # Check if this is a custom evaluator by checking if the base evaluator name is in built-ins
    # Remove numeric suffix (e.g., "ToolCallOrderEvaluator_1" -> "ToolCallOrderEvaluator")
    base_evaluator_name = re.sub(r'_\d+$', '', evaluator.id)

    if base_evaluator_name not in BUILT_IN_EVALUATORS:
        # For custom evaluators, determine the Python file and class name
        # This assumes custom evaluators follow a naming convention
        class_name = evaluator.__class__.__name__

        # Derive the filename from agent name (can be customized as needed)
        # For thermofisher_warranty custom evaluators, they're in thermofisher_warranty_evaluators.py
        file_name = f"{agent_name}_evaluators.py"

        spec["evaluatorSchema"] = f"file://{file_name}:{class_name}"

    return spec


def datapoint_to_evaluation(
    datapoint: Datapoint,
    eval_set_id: str,
    evaluator_refs: List[str],
    agent_name: str
) -> Dict[str, Any]:
    """Convert a Datapoint to an evaluation item in eval_set format (version 1.0).

    Args:
        datapoint: The Datapoint object to convert
        eval_set_id: The ID of the eval set this belongs to
        evaluator_refs: List of evaluator IDs referenced by this eval set
        agent_name: Name of the agent

    Returns:
        Dict containing the evaluation in UiPath format
    """
    # Map evaluation criteria to evaluator IDs
    # The criteria keys from datapoint are just evaluator names (e.g., "ContainsEvaluator")
    # But evaluator_refs now have agent prefix (e.g., "calculator_ContainsEvaluator")
    evaluation_criterias = {}
    for evaluator_name, criteria in datapoint.evaluation_criteria.items():
        # Add agent prefix to match the evaluator ID format
        agent_prefixed_id = f"{agent_name}_{evaluator_name}"
        if agent_prefixed_id in evaluator_refs:
            evaluation_criterias[agent_prefixed_id] = criteria

    return {
        "id": str(uuid.uuid4()),
        "name": datapoint.name,
        "inputs": datapoint.input,
        "expectedOutput": {},  # Can be populated if needed
        "simulationInstructions": datapoint.simulation_instructions,
        "expectedAgentBehavior": "",
        "simulateInput": False,
        "inputGenerationInstructions": "",
        "simulateTools": False,
        "toolsToSimulate": [],
        "evalSetId": eval_set_id,
        "createdAt": "2025-01-01T00:00:00.000Z",
        "updatedAt": "2025-01-01T00:00:00.000Z",
        "source": "manual",
        # New schema uses evaluationCriterias (plural) with evaluator IDs as keys
        "evaluationCriterias": evaluation_criterias,
    }


def export_evaluators(agent_name: str, output_dir: Path, only_supported: bool = False, include_llm_judge: bool = False) -> List[str]:
    """Export evaluator specs for an agent.

    Args:
        agent_name: Name of the agent (e.g., "calculator", "loan")
        output_dir: Directory to write evaluator specs to
        only_supported: If True, only export evaluators supported by the current PR

    Returns:
        List of evaluator IDs that were exported
    """
    # Currently supported evaluators in PR #685
    SUPPORTED_EVALUATORS = {
        "ContainsEvaluator",
        "ExactMatchEvaluator",
        "JsonSimilarityEvaluator",
        "LLMJudgeOutputEvaluator",
        "LLMJudgeStrictJSONSimilarityOutputEvaluator",
        "ToolCallOrderEvaluator",
        "ToolCallCountEvaluator",
        "ToolCallArgsEvaluator",
        "ToolCallOutputEvaluator",
        "LLMJudgeTrajectoryEvaluator",
        "LLMJudgeTrajectorySimulationEvaluator",
    }

    evaluators_getter = get_all_evaluators()[agent_name]
    evaluators = evaluators_getter(include_llm_judge)

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_ids = []

    for evaluator in evaluators:
        # Skip unsupported evaluators if only_supported flag is set
        # Extract base name by removing numeric suffix (e.g., "ToolCallOrderEvaluator_1" -> "ToolCallOrderEvaluator")
        base_name = re.sub(r'_\d+$', '', evaluator.name)

        if only_supported and base_name not in SUPPORTED_EVALUATORS:
            print(f"  ⊘ Skipped {evaluator.name} (not yet supported in PR)")
            continue
        spec = evaluator_to_spec(evaluator, agent_name)
        evaluator_id = spec["id"]
        evaluator_ids.append(evaluator_id)

        output_path = output_dir / f"evaluator-{evaluator_id}.json"

        with open(output_path, 'w') as f:
            json.dump(spec, f)

        print(f"  ✅ Exported evaluator: {evaluator_id}")

    return evaluator_ids


def export_eval_set(
    agent_name: str,
    evaluator_refs: List[str],
    output_dir: Path,
    small_set_size: int = 0
) -> None:
    """Export an agent's datapoints as a UiPath eval_set JSON file (version 1.0).

    Args:
        agent_name: Name of the agent (e.g., "calculator", "loan")
        evaluator_refs: List of evaluator IDs to reference
        output_dir: Directory to write the eval_set file to
    """
    agents = get_agents()
    agent = agents[agent_name]

    eval_set_id = str(uuid.uuid4())
    eval_set = {
        "version": "1.0",
        "id": eval_set_id,
        "name": f"{agent_name.title()} Eval Set",
        "evaluatorRefs": evaluator_refs,
        "evaluations": [
            datapoint_to_evaluation(dp, eval_set_id, evaluator_refs, agent_name)
            for dp in agent.datapoints
        ],
        "modelSettings": [],
        "createdAt": "2025-01-01T00:00:00.000Z",
        "updatedAt": "2025-01-01T00:00:00.000Z",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation-set-{agent_name}.json"

    with open(output_path, 'w') as f:
        json.dump(eval_set, f)

    print(small_set_size)
    if small_set_size > 0:
        small_eval_set_id = str(uuid.uuid4())
        eval_set["id"] = small_eval_set_id
        eval_set["name"] = f"{eval_set['name']} - Small"
        eval_set["evaluations"] = [
            datapoint_to_evaluation(dp, small_eval_set_id, evaluator_refs, agent_name)
            for dp in agent.datapoints[:small_set_size]
        ]
        output_path = output_dir / f"evaluation-set-{agent_name}-small.json"
        with open(output_path, 'w') as f:
            json.dump(eval_set, f)

    print(f"  ✅ Exported eval set with {len(agent.datapoints)} evaluations")


def export_agent(agent_name: str, base_dir: Path, only_supported: bool = False, include_llm_judge: bool = False, small_set_size: int = 0) -> None:
    """Export all evaluators and eval sets for a single agent.

    Args:
        agent_name: Name of the agent (e.g., "calculator", "loan")
        base_dir: Base evals directory
        only_supported: If True, only export evaluators supported by the current PR
    """
    print(f"\n📦 Exporting {agent_name} agent:")

    # Export evaluators
    evaluators_dir = base_dir / "evaluators"
    evaluator_ids = export_evaluators(agent_name, evaluators_dir, only_supported, include_llm_judge)

    # Export eval set
    eval_sets_dir = base_dir / "eval-sets"
    export_eval_set(agent_name, evaluator_ids, eval_sets_dir, small_set_size)

    print(f"✨ Completed {agent_name} agent export\n")


def main() -> None:
    """Export all agent evaluators and eval sets."""
    parser = argparse.ArgumentParser(description="Export evaluators and eval sets for agents")
    parser.add_argument(
        "--include_not_supported",
        action="store_true",
        help="Include evaluators not supported by the current PR (currently: ContainsEvaluator)"
    )
    parser.add_argument(
        "--exclude_llm_judge",
        action="store_true",
        help="Include LLM judge evaluators"
    )
    parser.add_argument(
        "--small_set_size",
        type=int,
        default=0,
        help="Size of the small eval set to export"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluations",
        help="Directory to write the eval_set file to"
    )
    args, _ = parser.parse_known_args()

    load_dotenv(find_dotenv())

    # Export to the standard location that uipath eval discovers
    base_dir = Path(__file__).parent.parent.parent / args.output_dir

    print("🚀 Starting export of evaluators and eval sets...")
    if args.include_not_supported:
        print("   (Including not supported evaluators)")

    print(f"\n📁 Files exported to: {base_dir.absolute()}")
    for agent_name in get_agents().keys():
        export_agent(agent_name, base_dir, only_supported=not args.include_not_supported, include_llm_judge=not args.exclude_llm_judge, small_set_size=args.small_set_size)
        print(f"       └── evaluation-set-{agent_name}.json")
        if args.small_set_size > 0:
            print(f"       └── evaluation-set-{agent_name}-small.json")
    print("\n✅ All exports completed!")

if __name__ == "__main__":
    main()
