from typing import List
from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from ..uipath_gym_types import AgentBaseClass, Datapoint
from ..tools import EscalationTool


class CalculatorInput(BaseModel):
    expression: str


class CalculatorOutput(BaseModel):
    answer: float


def get_datapoints() -> List[Datapoint]:
    """Get datapoints."""
    return [
        Datapoint(
            name="TestSimpleAddition",
            input={"expression": "how much is 2 + 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 7}},
                "ContainsEvaluator": {"search_text": "7"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 0)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 5}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 7}},
            },
        ),
        Datapoint(
            name="TestSimpleMultiplication",
            input={"expression": "how much is 2 * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 10.0}},
                "ContainsEvaluator": {"search_text": "10"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 0), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "multiply", "args": {"a": 2, "b": 5}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 10}},
            },
        ),
        Datapoint(
            name="TestSimpleSubtraction",
            input={"expression": "how much is 5 - 2"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 3.0}},
                "ContainsEvaluator": {"search_text": "3"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 0)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "add", "args": {"a": 5, "b": -2}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 3}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation1",
            input={"expression": "how much is 2 + 2 * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 12.0}},
                "ContainsEvaluator": {"search_text": "12"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 2, "b": 5}},
                    {"name": "add", "args": {"a": 2, "b": 10}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 12}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation2",
            input={"expression": "how much is 3 * 5 + 2 * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 25.0}},
                "ContainsEvaluator": {"search_text": "25"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 3, "b": 5}},
                    {"name": "multiply", "args": {"a": 2, "b": 5}},
                    {"name": "add", "args": {"a": 15, "b": 10}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 25}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation3",
            input={"expression": "how much is 3 * (5 + 5) * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 150.0}},
                "ContainsEvaluator": {"search_text": "150"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "multiply", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 5, "b": 5}},
                    {"name": "multiply", "args": {"a": 3, "b": 10}},
                    {"name": "multiply", "args": {"a": 30, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 150}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation4",
            input={"expression": "how much is 2 + 3 * 4"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 14.0}},
                "ContainsEvaluator": {"search_text": "14"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 3, "b": 4}},
                    {"name": "add", "args": {"a": 2, "b": 12}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 14}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation5",
            input={"expression": "how much is (2 + 3) * (4 + 1)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 25.0}},
                "ContainsEvaluator": {"search_text": "25"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 2), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 2, "b": 3}},
                    {"name": "add", "args": {"a": 4, "b": 1}},
                    {"name": "multiply", "args": {"a": 5, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 25}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation6",
            input={"expression": "how much is 10 * 0.5 + 3"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 8.0}},
                "ContainsEvaluator": {"search_text": "8"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 10, "b": 0.5}},
                    {"name": "add", "args": {"a": 5, "b": 3}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 8}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation7",
            input={"expression": "how much is 5 * 5 + (-10)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 15.0}},
                "ContainsEvaluator": {"search_text": "15"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 5, "b": 5}},
                    {"name": "add", "args": {"a": 25, "b": -10}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 15}},
            },
        ),
        Datapoint(
            name="TestCompoundOperation8",
            input={"expression": "how much is (8 + (-3)) * (2 + 2)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 20.0}},
                "ContainsEvaluator": {"search_text": "20"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 2), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 8, "b": -3}},
                    {"name": "add", "args": {"a": 2, "b": 2}},
                    {"name": "multiply", "args": {"a": 5, "b": 4}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 20}},
            },
        ),
        Datapoint(
            name="TestNestedParentheses",
            input={"expression": "how much is (2 * (3 + 4)) * (1 + 1)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 28.0}},
                "ContainsEvaluator": {"search_text": "28"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 2), "multiply": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "multiply", "add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 3, "b": 4}},
                    {"name": "multiply", "args": {"a": 2, "b": 7}},
                    {"name": "add", "args": {"a": 1, "b": 1}},
                    {"name": "multiply", "args": {"a": 14, "b": 2}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 28}},
            },
        ),
        Datapoint(
            name="TestMultipleNegatives",
            input={"expression": "how much is (-2) * (-3) + (-5)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 1.0}},
                "ContainsEvaluator": {"search_text": "1"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": -2, "b": -3}},
                    {"name": "add", "args": {"a": 6, "b": -5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 1}},
            },
        ),
        Datapoint(
            name="TestDecimalParentheses",
            input={"expression": "how much is (0.5 + 1.5) * 2.5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 5.0}},
                "ContainsEvaluator": {"search_text": "5"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 0.5, "b": 1.5}},
                    {"name": "multiply", "args": {"a": 2, "b": 2.5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 5}},
            },
        ),
        Datapoint(
            name="TestMissingNumberSimple",
            input={"expression": "how much is x + 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 22.0}},
                "ContainsEvaluator": {"search_text": "22"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 17, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 22}},
            },
        ),
        Datapoint(
            name="TestMissingNumberMedium",
            input={"expression": "how much is x + y * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 102.0}},
                "ContainsEvaluator": {"search_text": "102"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 2), "multiply": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "escalation", "multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 17, "b": 5}},
                    {"name": "add", "args": {"a": 17, "b": 85}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 102}},
            },
        ),
        Datapoint(
            name="TestMissingNumberComplex",
            input={"expression": "how much is x + y * z + 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 311.0}},
                "ContainsEvaluator": {"search_text": "311"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 3), "multiply": ("=", 1), "add": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "escalation", "escalation", "multiply", "add", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 17, "b": 17}},
                    {"name": "add", "args": {"a": 17, "b": 289}},
                    {"name": "add", "args": {"a": 306, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 311}},
            },
        ),
    ]


def get_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            func=lambda a, b: a + b,
            name="add",
            description="Add two numbers together",
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
            description="Multiply two numbers together",
            args_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
        ),
        EscalationTool(
            name="escalation",
            description="Escalate the issue to a human agent if you are missing a number. If you are missing more numbers, call this for each missing number.",
            assign_to="random@uipath.com",
            return_message="The missing number is 17. Continue with the evaluation.",
        ),
    ]



def get_calculator_agent() -> AgentBaseClass:
    return (
        AgentBaseClass(
            system_prompt="You are a calculator agent. You are given a mathematical expression and you need to evaluate it. If any number is missing, use the escalation tool to get the value for it.",
            user_prompt="Calculate the result of: {expression}.",
            input_schema=CalculatorInput,
            output_schema=CalculatorOutput,
            tools=get_tools(),
            datapoints=get_datapoints(),
    )
)
