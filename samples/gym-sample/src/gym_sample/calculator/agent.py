from typing import List
from langchain_core.tools import BaseTool, StructuredTool
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
                "ExactMatchEvaluator": {"expected_output": {"answer": 7.0}},
                "ContainsEvaluator": {"search_text": "7"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 7.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 0)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 5}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 7}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 2 and 5 to get 7."},
            },
        ),
        Datapoint(
            name="TestSimpleMultiplication",
            input={"expression": "how much is 2 * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 10.0}},
                "ContainsEvaluator": {"search_text": "10"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 10.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 0), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "multiply", "args": {"a": 2, "b": 5}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 10}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 2 and 5 to get 10."},
            },
        ),
        Datapoint(
            name="TestSimpleSubtraction",
            input={"expression": "how much is 5 - 2"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 3.0}},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 3.0}},
                "ContainsEvaluator": {"search_text": "3"},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 0)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "add", "args": {"a": 5, "b": -2}}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 3}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 5 and -2 to get 3."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation1",
            input={"expression": "how much is 2 + 2 * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 12.0}},
                "ContainsEvaluator": {"search_text": "12"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 12.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 2, "b": 5}},
                    {"name": "add", "args": {"a": 2, "b": 10}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 12}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 2 and 5 to get 10, then use the add tool to add 2 and 10 to get 12."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation2",
            input={"expression": "how much is 3 * 5 + 2 * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 25.0}},
                "ContainsEvaluator": {"search_text": "25"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 25.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 3, "b": 5}},
                    {"name": "multiply", "args": {"a": 2, "b": 5}},
                    {"name": "add", "args": {"a": 15, "b": 10}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 25}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 3 and 5 to get 15, then use the multiply tool to multiply 2 and 5 to get 10, then use the add tool to add 15 and 10 to get 25."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation3",
            input={"expression": "how much is 3 * (5 + 5) * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 150.0}},
                "ContainsEvaluator": {"search_text": "150"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 150.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "multiply", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 5, "b": 5}},
                    {"name": "multiply", "args": {"a": 3, "b": 10}},
                    {"name": "multiply", "args": {"a": 30, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 150}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 5 and 5 to get 10, then use the multiply tool to multiply 3 and 10 to get 30, then use the multiply tool to multiply 30 and 5 to get 150."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation4",
            input={"expression": "how much is 2 + 3 * 4"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 14.0}},
                "ContainsEvaluator": {"search_text": "14"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 14.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 3, "b": 4}},
                    {"name": "add", "args": {"a": 2, "b": 12}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 14}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 3 and 4 to get 12, then use the add tool to add 2 and 12 to get 14."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation5",
            input={"expression": "how much is (2 + 3) * (4 + 1)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 25.0}},
                "ContainsEvaluator": {"search_text": "25"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 25.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 2), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 2, "b": 3}},
                    {"name": "add", "args": {"a": 4, "b": 1}},
                    {"name": "multiply", "args": {"a": 5, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 25}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 2 and 3 to get 5, then use the add tool to add 4 and 1 to get 5, then use the multiply tool to multiply 5 and 5 to get 25."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation6",
            input={"expression": "how much is 10 * 0.5 + 3"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 8.0}},
                "ContainsEvaluator": {"search_text": "8"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 8.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 10, "b": 0.5}},
                    {"name": "add", "args": {"a": 5, "b": 3}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 8}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 10 and 0.5 to get 5, then use the add tool to add 5 and 3 to get 8."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation7",
            input={"expression": "how much is 5 * 5 + (-10)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 15.0}},
                "ContainsEvaluator": {"search_text": "15"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 15.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 5, "b": 5}},
                    {"name": "add", "args": {"a": 25, "b": -10}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 15}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 5 and 5 to get 25, then use the add tool to add 25 and -10 to get 15."},
            },
        ),
        Datapoint(
            name="TestCompoundOperation8",
            input={"expression": "how much is (8 + (-3)) * (2 + 2)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 20.0}},
                "ContainsEvaluator": {"search_text": "20"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 20.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 2), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 8, "b": -3}},
                    {"name": "add", "args": {"a": 2, "b": 2}},
                    {"name": "multiply", "args": {"a": 5, "b": 4}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 20}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 8 and -3 to get 5, then use the add tool to add 2 and 2 to get 4, then use the multiply tool to multiply 5 and 4 to get 20."},
            },
        ),
        Datapoint(
            name="TestNestedParentheses",
            input={"expression": "how much is (2 * (3 + 4)) * (1 + 1)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 28.0}},
                "ContainsEvaluator": {"search_text": "28"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 28.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 2), "multiply": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "multiply", "add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 3, "b": 4}},
                    {"name": "multiply", "args": {"a": 2, "b": 7}},
                    {"name": "add", "args": {"a": 1, "b": 1}},
                    {"name": "multiply", "args": {"a": 14, "b": 2}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 28}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 3 and 4 to get 7, then use the multiply tool to multiply 2 and 7 to get 14, then use the add tool to add 1 and 1 to get 2, then use the multiply tool to multiply 14 and 2 to get 28."},
            },
        ),
        Datapoint(
            name="TestMultipleNegatives",
            input={"expression": "how much is (-2) * (-3) + (-5)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 1.0}},
                "ContainsEvaluator": {"search_text": "1"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 1.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": -2, "b": -3}},
                    {"name": "add", "args": {"a": 6, "b": -5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 1}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply -2 and -3 to get 6, then use the add tool to add 6 and -5 to get 1."},
            },
        ),
        Datapoint(
            name="TestDecimalParentheses",
            input={"expression": "how much is (0.5 + 1.5) * 2.5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 5.0}},
                "ContainsEvaluator": {"search_text": "5"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 5.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 0.5, "b": 1.5}},
                    {"name": "multiply", "args": {"a": 2, "b": 2.5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 5}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 0.5 and 1.5 to get 2, then use the multiply tool to multiply 2 and 2.5 to get 5."},
            },
        ),
        Datapoint(
            name="TestMissingNumberSimple",
            input={"expression": "how much is x + 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 22.0}},
                "ContainsEvaluator": {"search_text": "22"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 22.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "add", "args": {"a": 17, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 22}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the escalation tool to escalate 17 and 5 to get 22."},
            },
        ),
        Datapoint(
            name="TestMissingNumberMedium",
            input={"expression": "how much is x + y * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 102.0}},
                "ContainsEvaluator": {"search_text": "102"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 102.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 2), "multiply": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "escalation", "multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 17, "b": 5}},
                    {"name": "add", "args": {"a": 17, "b": 85}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 102}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the escalation tool to escalate 17 and 5 to get 22, then use the escalation tool to escalate 22 and 5 to get 27, then use the multiply tool to multiply 27 and 17 to get 459, then use the add tool to add 459 and 85 to get 544."},
            },
        ),
        Datapoint(
            name="TestMissingNumberComplex",
            input={"expression": "how much is x + y * z + 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 311.0}},
                "ContainsEvaluator": {"search_text": "311"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 311.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 3), "multiply": ("=", 1), "add": ("=", 2)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "escalation", "escalation", "multiply", "add", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 17, "b": 17}},
                    {"name": "add", "args": {"a": 17, "b": 289}},
                    {"name": "add", "args": {"a": 306, "b": 5}}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 311}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the escalation tool to escalate 17 and 17 to get 289, then use the escalation tool to escalate 289 and 17 to get 306, then use the escalation tool to escalate 306 and 5 to get 311."},
            },
        ),
        Datapoint(
            name="TestToolCallOutput1",
            input={"expression": "how much is 10 + 15"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 25.0}},
                "ContainsEvaluator": {"search_text": "25"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 25.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "add", "args": {"a": 10, "b": 15}}]},
                "ToolCallOutputEvaluator": {"tool_outputs": [{"name": "add", "output": "25"}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 25}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 10 and 15 to get 25."},
            },
        ),
        Datapoint(
            name="TestToolCallOutput2",
            input={"expression": "how much is 6 * 7"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 42.0}},
                "ContainsEvaluator": {"search_text": "42"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 42.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply"]},
                "ToolCallArgsEvaluator": {"tool_calls": [{"name": "multiply", "args": {"a": 6, "b": 7}}]},
                "ToolCallOutputEvaluator": {"tool_outputs": [{"name": "multiply", "output": "42"}]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 42}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 6 and 7 to get 42."},
            },
        ),
        Datapoint(
            name="TestToolCallOutput3",
            input={"expression": "how much is 100 + 50 * 2"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 200.0}},
                "ContainsEvaluator": {"search_text": "200"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 200.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"multiply": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "ToolCallArgsEvaluator": {"tool_calls": [
                    {"name": "multiply", "args": {"a": 50, "b": 2}},
                    {"name": "add", "args": {"a": 100, "b": 100}}
                ]},
                "ToolCallOutputEvaluator": {"tool_outputs": [
                    {"name": "multiply", "output": "100"},
                    {"name": "add", "output": "200"}
                ]},
                "LLMJudgeOutputEvaluator": {"expected_output": {"answer": 200}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 50 and 2 to get 100, then use the add tool to add 100 and 100 to get 200."},
            },
        ),
        Datapoint(
            name="TestStrictJSONSimilarity1",
            input={"expression": "how much is 12 + 13"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 25.0}},
                "ContainsEvaluator": {"search_text": "25"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 25.0}},
                "LLMJudgeStrictJSONSimilarityOutputEvaluator": {"expected_output": {"answer": 25.0}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the add tool to add 12 and 13 to get 25."},
            },
        ),
        Datapoint(
            name="TestStrictJSONSimilarity2",
            input={"expression": "how much is 9 * 9"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 81.0}},
                "ContainsEvaluator": {"search_text": "81"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 81.0}},
                "LLMJudgeStrictJSONSimilarityOutputEvaluator": {"expected_output": {"answer": 81.0}},
                "LLMJudgeTrajectoryEvaluator": {"expected_agent_behavior": "The agent should use the multiply tool to multiply 9 and 9 to get 81."},
            },
        ),
        Datapoint(
            name="TestTrajectory1",
            input={"expression": "how much is 4 + 5 * 3"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 19.0}},
                "ContainsEvaluator": {"search_text": "19"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 19.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"multiply": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "add"]},
                "LLMJudgeTrajectoryEvaluator": {
                    "expected_agent_behavior": "The agent should follow the order of operations, first multiplying 5 and 3 to get 15, then adding 4 to get 19."
                },
            },
        ),
        Datapoint(
            name="TestTrajectory2",
            input={"expression": "how much is (1 + 2) * (3 + 4)"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 21.0}},
                "ContainsEvaluator": {"search_text": "21"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 21.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"add": ("=", 2), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["add", "add", "multiply"]},
                "LLMJudgeTrajectoryEvaluator": {
                    "expected_agent_behavior": "The agent should evaluate parentheses first: add 1 and 2 to get 3, add 3 and 4 to get 7, then multiply 3 and 7 to get 21."
                },
            },
        ),
        Datapoint(
            name="TestTrajectory3",
            input={"expression": "how much is 2 * 3 + 4 * 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 26.0}},
                "ContainsEvaluator": {"search_text": "26"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 26.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"multiply": ("=", 2), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["multiply", "multiply", "add"]},
                "LLMJudgeTrajectoryEvaluator": {
                    "expected_agent_behavior": "The agent should perform multiplications first (2*3=6 and 4*5=20), then add the results (6+20=26)."
                },
            },
        ),
        Datapoint(
            name="TestTrajectorySimulation1",
            input={"expression": "how much is a + 10"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 27.0}},
                "ContainsEvaluator": {"search_text": "27"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 27.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "add"]},
                "LLMJudgeTrajectorySimulationEvaluator": {
                    "expected_agent_behavior": "The agent should recognize that 'a' is an unknown variable, escalate to get its value (17), then add 10 to get 27."
                },
            },
            simulation_instructions="When the agent asks for the value of 'a', respond with 17.",
        ),
        Datapoint(
            name="TestTrajectorySimulation2",
            input={"expression": "how much is m * n + 5"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 294.0}},
                "ContainsEvaluator": {"search_text": "294"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 294.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 2), "multiply": ("=", 1), "add": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "escalation", "multiply", "add"]},
                "LLMJudgeTrajectorySimulationEvaluator": {
                    "expected_agent_behavior": "The agent should identify two missing variables (m and n), escalate twice to get their values (both 17), multiply them to get 289, then add 5 to get 294."
                },
            },
            simulation_instructions="When the agent asks for missing values, respond with 17 for each variable.",
        ),
        Datapoint(
            name="TestTrajectorySimulation3",
            input={"expression": "how much is (p + q) * r"},
            evaluation_criteria={
                "ExactMatchEvaluator": {"expected_output": {"answer": 578.0}},
                "ContainsEvaluator": {"search_text": "578"},
                "JsonSimilarityEvaluator": {"expected_output": {"answer": 578.0}},
                "ToolCallCountEvaluator": {"tool_calls_count": {"escalation": ("=", 3), "add": ("=", 1), "multiply": ("=", 1)}},
                "ToolCallOrderEvaluator": {"tool_calls_order": ["escalation", "escalation", "escalation", "add", "multiply"]},
                "LLMJudgeTrajectorySimulationEvaluator": {
                    "expected_agent_behavior": "The agent should recognize three missing variables (p, q, r), escalate three times to get their values (all 17), add p and q (17+17=34), then multiply by r (34*17=578)."
                },
            },
            simulation_instructions="When the agent asks for missing values, respond with 17 for each variable (p, q, and r).",
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
