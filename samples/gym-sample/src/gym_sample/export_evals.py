"""Export AgentBaseClass evaluators and datapoints to UiPath eval format.

This script exports:
1. Evaluator specs in the new schema format (version 1.0) to evals/evaluators/
2. Eval sets that reference those evaluators to evals/eval-sets/
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from gym_sample.graph import get_agents, get_all_evaluators


def evaluator_to_spec(evaluator: Any, agent_name: str) -> Dict[str, Any]:
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
    evaluator_id = f"{agent_name}_{evaluator.name}"

    # Use the evaluator_config (Pydantic model) and dump with aliases (camelCase)
    evaluator_config = evaluator.evaluator_config.model_dump(
        by_alias=True,
        exclude_none=True,
        exclude_unset=False  # Include defaults like targetOutputKey
    )

    return {
        "version": "1.0",
        "id": evaluator_id,
        "description": f"{evaluator.name} for {agent_name} agent",
        "evaluatorTypeId": evaluator_type_id,
        "evaluatorConfig": evaluator_config,
    }


def datapoint_to_evaluation(
    datapoint: Any,
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


def export_evaluators(agent_name: str, output_dir: Path, only_supported: bool = False) -> List[str]:
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
        "ContainsEvaluator"
    }

    evaluators_getter = get_all_evaluators()[agent_name]
    evaluators = evaluators_getter(False)  # Export without LLM judges by default

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_ids = []

    for evaluator in evaluators:
        # Skip unsupported evaluators if only_supported flag is set
        if only_supported and evaluator.name not in SUPPORTED_EVALUATORS:
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
    output_dir: Path
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

    print(f"  ✅ Exported eval set with {len(agent.datapoints)} evaluations")


def export_agent(agent_name: str, base_dir: Path, only_supported: bool = False) -> None:
    """Export all evaluators and eval sets for a single agent.

    Args:
        agent_name: Name of the agent (e.g., "calculator", "loan")
        base_dir: Base evals directory
        only_supported: If True, only export evaluators supported by the current PR
    """
    print(f"\n📦 Exporting {agent_name} agent:")

    # Export evaluators
    evaluators_dir = base_dir / "evaluators"
    evaluator_ids = export_evaluators(agent_name, evaluators_dir, only_supported)

    # Export eval set
    eval_sets_dir = base_dir / "eval-sets"
    export_eval_set(agent_name, evaluator_ids, eval_sets_dir)

    print(f"✨ Completed {agent_name} agent export\n")


def main() -> None:
    """Export all agent evaluators and eval sets."""
    parser = argparse.ArgumentParser(description="Export evaluators and eval sets for agents")
    parser.add_argument(
        "--only-supported",
        action="store_true",
        help="Only export evaluators supported by the current PR (currently: ContainsEvaluator)"
    )
    args, _ = parser.parse_known_args()

    # Export to the standard location that uipath eval discovers
    base_dir = Path(__file__).parent.parent.parent / "evals"

    print("🚀 Starting export of evaluators and eval sets...")
    if args.only_supported:
        print("   (Only exporting supported evaluators)")

    for agent_name in ["calculator", "loan"]:
        export_agent(agent_name, base_dir, only_supported=args.only_supported)

    print("✅ All exports completed!")
    print(f"\n📁 Files exported to: {base_dir.absolute()}")
    print("   ├── evaluators/")
    print("   │   ├── evaluator-calculator_*.json")
    print("   │   └── evaluator-loan_*.json")
    print("   └── eval-sets/")
    print("       ├── evaluation-set-calculator.json")
    print("       └── evaluation-set-loan.json")


if __name__ == "__main__":
    main()
