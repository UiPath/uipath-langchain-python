from typing import List
from gym_sample.uipath_gym_types import Datapoint


def get_datapoints() -> List[Datapoint]:
    """Get datapoints."""
    return [
        Datapoint(
            input={
                "expression": "15.0 + 7.0 * 3.0"
            },
            evaluation_criteria={
                "exact_match": {"answer": 36.0},
                "tool_call_order": ["multiply", "add"],
                "tool_call_count": {"multiply": "ge:1", "add": "ge:1"},
                "tool_call_arguments": [{"name": "multiply", "args": {"a": 7., "b":3.}}, {"name": "add", "args": {"a": 15., "b": 21.}}],
                "tool_call_output": [{"name": "multiply", "output": "21.0"}, {"name": "add", "output": "36.0"}],
                "llm_judge": {"answer": 36.0},
                "llm_judge_strict_json_similarity": {"answer": 36.0},
                "llm_judge_trajectory": "The agent should have called the multiply tool with the arguments 7.0 and 3.0, and the add tool with the arguments 15.0 and 21.0.",
                "llm_judge_simulation_trajectory": "The agent should have called the multiply tool with the arguments 7.0 and 3.0, and the add tool with the arguments 15.0 and 21.0.",
            },
            simulation_instructions="Tool multiply should return 21.0 and tool add should return 36.0.",
        ),
        Datapoint(
            input={
                "expression": "20 + 5 * 2.0"
            },
            evaluation_criteria={
                "exact_match": {"answer": 30.0},
                "tool_call_order": ["multiply", "add"],
                "tool_call_count": {"multiply": "ge:1", "add": "ge:1"},
                "tool_call_arguments": [{"name": "multiply", "args": {"a": 5., "b":2.}}, {"name": "add", "args": {"a": 20., "b": 10.}}],
                "tool_call_output": [{"name": "multiply", "output": "10.0"}, {"name": "add", "output": "30.0"}],
                "llm_judge": {"answer": 30.0},
                "llm_judge_strict_json_similarity": {"answer": 30.0},
                "llm_judge_trajectory": "The agent should have called the multiply tool with the arguments 5.0 and 2.0, and the add tool with the arguments 20.0 and 10.0.",
                "llm_judge_simulation_trajectory": "The agent should have called the multiply tool with the arguments 5.0 and 2.0, and the add tool with the arguments 20.0 and 10.0.",
            },
            simulation_instructions="Tool multiply should return 10.0 and tool add should return 30.0.",
        ),
    ]
