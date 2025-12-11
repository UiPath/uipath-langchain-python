from typing import List

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from uipath._cli._evals._models._evaluation_set import EvaluationItem, EvaluationSet

from gym_sample.tools import EscalationTool
from gym_sample.uipath_gym_types import AgentBaseClass


class CalculatorInput(BaseModel):
    expression: str


class CalculatorOutput(BaseModel):
    answer: float


def get_coded_evaluation_set() -> EvaluationSet:
    """Get EvaluationItems."""
    evaluation_set = EvaluationSet(
        id="calculator_evaluation_set",
        name="Calculator Evaluation Set",
        version="1.0",
        evaluator_refs=[
            "calculator_ExactMatchEvaluator",
            "calculator_ContainsEvaluator",
            "calculator_JsonSimilarityEvaluator",
            "calculator_ToolCallOrderEvaluator",
            "calculator_ToolCallCountEvaluator",
            "calculator_ToolCallArgsEvaluator",
            "calculator_ToolCallOutputEvaluator",
            "calculator_LLMJudgeOutputEvaluator",
            "calculator_LLMJudgeTrajectoryEvaluator",
            "calculator_LLMJudgeTrajectorySimulationEvaluator",
        ],
        evaluations=[
            EvaluationItem(
                id="calculator_TestSimpleAddition",
                name="TestSimpleAddition",
                inputs={"expression": "how much is 2 + 5"},
                evaluationCriterias={
                    "calculator_ContainsEvaluator": {"search_text": "7"},
                    "calculator_JsonSimilarityEvaluator": {
                        "expected_output": {"answer": 7.0}
                    },
                    "calculator_ToolCallCountEvaluator": {
                        "tool_calls_count": {"add": ("=", 1), "multiply": ("=", 0)}
                    },
                    "calculator_ToolCallOrderEvaluator": {"tool_calls_order": ["add"]},
                    "calculator_ToolCallArgsEvaluator": {
                        "tool_calls": [{"name": "add", "args": {"a": 2, "b": 5}}]
                    },
                    "calculator_LLMJudgeOutputEvaluator": {
                        "expected_output": {"answer": 7}
                    },
                    "calculator_LLMJudgeTrajectoryEvaluator": {
                        "expected_agent_behavior": "The agent should use the add tool to add 2 and 5 to get 7."
                    },
                },
            ),
            EvaluationItem(
                id="calculator_TestSimpleMultiplication",
                name="TestSimpleMultiplication",
                inputs={"expression": "how much is 2 * 5"},
                evaluationCriterias={
                    "calculator_ExactMatchEvaluator": {
                        "expected_output": {"answer": 10.0}
                    },
                    "calculator_ContainsEvaluator": {"search_text": "10"},
                    "calculator_JsonSimilarityEvaluator": {
                        "expected_output": {"answer": 10.0}
                    },
                    "calculator_ToolCallCountEvaluator": {
                        "tool_calls_count": {"add": ("=", 0), "multiply": ("=", 1)}
                    },
                    "calculator_ToolCallOrderEvaluator": {
                        "tool_calls_order": ["multiply"]
                    },
                    "calculator_ToolCallArgsEvaluator": {
                        "tool_calls": [{"name": "multiply", "args": {"a": 2, "b": 5}}]
                    },
                    "calculator_LLMJudgeOutputEvaluator": {
                        "expected_output": {"answer": 10}
                    },
                    "calculator_LLMJudgeTrajectoryEvaluator": {
                        "expected_agent_behavior": "The agent should use the multiply tool to multiply 2 and 5 to get 10."
                    },
                },
            ),
        ],
    )
    return evaluation_set


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
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
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
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
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
    return AgentBaseClass(
        system_prompt="You are a calculator agent. You are given a mathematical expression and you need to evaluate it. If any number is missing, use the escalation tool to get the value for it.",
        user_prompt="Calculate the result of: {expression}.",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        tools=get_tools(),
        evaluation_set=get_coded_evaluation_set(),
        # evaluation_set=get_evaluation_set(
        #     Path(__file__).parent / "eval-sets" / "evaluation-set-calculator.json"
        # ),
    )
