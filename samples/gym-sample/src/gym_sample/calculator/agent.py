from typing import List
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from ..uipath_gym_types import AgentBaseClass, Datapoint
from langchain_core.tools import StructuredTool


class CalculatorInput(BaseModel):
    expression: str


class CalculatorOutput(BaseModel):
    answer: float


def get_datapoints() -> List[Datapoint]:
    """Get datapoints."""
    return [
        Datapoint(
            name="datapoint_1",
            input={
                "expression": "15.0 + 7.0 * 3.0"
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 36.0}},
                "ContainsEvaluator": {"search_text": "36"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 36.0}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"multiply": (">=", 1), "add": (">=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "multiply", "args": {"a": 7., "b":3.}}, {"name": "add", "args": {"a": 15., "b": 21.}}]},
                "ToolCallOutputEvaluator": {"tool_outputs": [{"name": "multiply", "output": "21.0"}, {"name": "add", "output": "36.0"}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 36.0}},
                "LLMJudgeStrictJSONSimilarityOutputEvaluator": {"expected_output": {"answer": 36.0}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should have called the multiply tool with the arguments 7.0 and 3.0, and the add tool with the arguments 15.0 and 21.0."},
                "LLMJudgeSimulationEvaluator": {"expected_agent_behavior": "The agent should have called the multiply tool with the arguments 7.0 and 3.0, and the add tool with the arguments 15.0 and 21.0."},
            },
            simulation_instructions="Tool multiply should return 21.0 and tool add should return 36.0.",
        ),
        Datapoint(
            name="datapoint_2",
            input={
                "expression": "20 + 5 * 2.0"
            },
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 30.0}},
                "ContainsEvaluator": {"search_text": "30"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 30.0}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallCountEvaluator": {"tool_calls_count": {"multiply": (">=", 1), "add": (">=", 1)}},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "multiply", "args": {"a": 5., "b":2.}}, {"name": "add", "args": {"a": 20., "b": 10.}}]},
                "ToolCallOutputEvaluator": {"tool_outputs": [{"name": "multiply", "output": "10.0"}, {"name": "add", "output": "30.0"}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 30.0}},
                "LLMJudgeStrictJSONSimilarityOutputEvaluator": {"expected_output": {"answer": 30.0}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should have called the multiply tool with the arguments 5.0 and 2.0, and the add tool with the arguments 20.0 and 10.0."},
                "LLMJudgeSimulationEvaluator": {"expected_agent_behavior": "The agent should have called the multiply tool with the arguments 5.0 and 2.0, and the add tool with the arguments 20.0 and 10.0."},
            },
            simulation_instructions="Tool multiply should return 10.0 and tool add should return 30.0.",
        ),
    ]


def get_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            func=lambda a, b: a + b,
            name="add",
            description="Add two numbers",
            args_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
        ),
        StructuredTool.from_function(
            func=lambda a, b: a * b,
            name="multiply",
            description="Multiply two numbers",
            args_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
    )]



def get_calculator_agent() -> AgentBaseClass:
    return (
        AgentBaseClass(
            system_prompt="You are a calculator agent. You can perform mathematical operations using the available tools. When you have completed the calculation, use the end_execution tool to provide your final result with a score (0.0 to 1.0 representing confidence) and observations about the calculation process.",
            user_prompt="Calculate the result of: {expression}.",
            input_schema=CalculatorInput,
            output_schema=CalculatorOutput,
            tools=get_tools(),
            datapoints=get_datapoints(),
    )
)
