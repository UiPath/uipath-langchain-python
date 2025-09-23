
from collections.abc import Mapping
from datetime import datetime
import json
from typing import Any, Dict, List, Set

from opentelemetry.sdk.trace import ReadableSpan
from uipath.eval.evaluators.base_evaluator import AgentExecution as BaseAgentExecution


def extract_tool_calls_names(spans: List[ReadableSpan]) -> List[str]:
    """Extract the tool call names from execution spans IN ORDER.

    Args:
        spans: List of ReadableSpan objects from agent execution.

    Returns:
        List of tool names in the order they were called.
    """
    tool_calls_names = []

    for span in spans:
        # Check for tool.name attribute first
        if span.attributes and (tool_name := span.attributes.get('tool.name')):
            tool_calls_names.append(tool_name)

    return tool_calls_names


def extract_tool_calls(spans: List[ReadableSpan]) -> List[Dict[str, Any]]:
    """Extract the tool calls from execution spans with their arguments.

    Args:
        spans: List of ReadableSpan objects from agent execution.

    Returns:
        Dict of tool calls with their arguments.
    """
    tool_calls = []

    for span in spans:
        if span.attributes and (tool_name := span.attributes.get('tool.name')):
            try:
                input_value = span.attributes.get('input.value', '{}')
                # Ensure input_value is a string before parsing
                if isinstance(input_value, str):
                    arguments = json.loads(input_value.replace("'", '"'))
                else:
                    arguments = {}
                tool_calls.append({"name": tool_name, "args": arguments})
            except json.JSONDecodeError:
                # Handle case where input.value is not valid JSON
                tool_calls.append({"name": tool_name, "args": {}})

    return tool_calls


def tool_calls_order_score(
    actual_tool_calls_names: List[str], expected_tool_calls_names: List[str], strict: bool = False
) -> float:
    """
    The function calculates the longest common subsequence between the actual tool calls
    and the expected tool calls and returns the ratio of the LCS length to the number of
    expected calls.

    Args:
        actual_tool_calls_names: List of tool names in the actual order
        expected_tool_calls_names: List of tool names in the expected order
        strict: If True, the function will return 0 if the actual calls do not match the expected calls

    Returns:
        float: Ratio of the LCS length to the number of expected
    """
    if (
        not expected_tool_calls_names
        and not actual_tool_calls_names
        or expected_tool_calls_names == actual_tool_calls_names
    ):
        return 1.0
    elif (
        not expected_tool_calls_names
        or not actual_tool_calls_names
        or strict
        and actual_tool_calls_names != expected_tool_calls_names
    ):
        return 0.0

    # Calculate LCS with DP + memory efficient
    m, n = len(actual_tool_calls_names), len(expected_tool_calls_names)
    min_length, max_length = min(m, n), max(m, n)
    dp = [[0] * (min_length + 1) for _ in range(2)]

    aux_actual, aux_expected = (
        (actual_tool_calls_names, expected_tool_calls_names)
        if m >= n
        else (expected_tool_calls_names, actual_tool_calls_names)
    )

    for i in range(1, max_length + 1):
        for j in range(1, min_length + 1):
            if aux_actual[i - 1] == aux_expected[j - 1]:
                dp[1][j] = dp[0][j - 1] + 1
            else:
                dp[1][j] = max(dp[0][j], dp[1][j - 1])
        dp[0] = dp[1]

    lcs_length = dp[-1][-1]
    return lcs_length / n


def tool_calls_count_score(actual_tool_calls_count: Mapping[str, int | str], expected_tool_calls_count: Mapping[str, int | str], strict: bool = False) -> float:
    """
    Check if the expected tool calls are correctly called, where expected args must be a subset of actual args.
    It does not check the order of the tool calls!
    """
    if (
        not expected_tool_calls_count
        and not actual_tool_calls_count
    ):
        return 1.0
    elif (
        not expected_tool_calls_count
        or not actual_tool_calls_count
    ):
        return 0.0

    score = 0.0
    for tool_name, expected_count in expected_tool_calls_count.items():
        actual_count = actual_tool_calls_count.get(tool_name, 0.0)
        if isinstance(expected_count, str):
            try:
                comparator, expected_count = expected_count.split(":")
            except (IndexError, ValueError) as e:
                raise ValueError(f"Wrong format for expected count for tool {tool_name}: {expected_count}") from e
            comparator = f"__{comparator}__"
        else:
            comparator = "__eq__"
        to_add = float(getattr(actual_count, comparator)(int(expected_count)))
        if strict:
            if to_add == 0.0:
                return 0.0
        else:
            score += to_add
    return score / len(expected_tool_calls_count)


def tool_args_score(actual_tool_calls: List[Dict[str, Any]], expected_tool_calls: List[Dict[str, Any]], strict: bool = False, subset: bool = False) -> float:
    """
    Check if the expected tool calls are correctly called, where expected args must be a subset of actual args.
    It does not check the order of the tool calls!

    Arguments:
        actual_tool_calls (list[Dict[str, Any]]): List of actual tool calls in the format of {"name": str, "args": Dict[str, Any]}
        expected_tool_calls (list[Dict[str, Any]]): List of expected tool calls in the format of {"name": str, "args": Dict[str, Any]}
        strict (bool): If True, the function will return 0 if not all expected tool calls are matched
        subset (bool): If True, the function will check if the expected args are a subset of the actual args

    Returns:
        float: Score based on the number of matches
    """
    cnt = 0
    visited: set[int] = set()

    for expected_tool_call in expected_tool_calls:
        for idx, call in enumerate(actual_tool_calls):
            if call.get('name') == expected_tool_call.get('name') and idx not in visited:
                # Check arguments based on mode
                if subset:
                    # Subset mode: safely check if all expected args exist and match
                    args_check = lambda k, v: k in call.get('args', {}) and call.get('args', {})[k] == v
                    validator_check = lambda k, validator: k not in call.get('args', {}) or validator(
                        call.get('args', {})[k]
                    )
                else:
                    # Exact mode: direct access (may raise KeyError)
                    args_check = lambda k, v: call.get('args', {})[k] == v
                    validator_check = lambda k, validator: validator(call.get('args', {})[k])

                try:
                    args_match = all(args_check(k, v) for k, v in expected_tool_call.get('args', {}).items())
                    validators_match = True
                    if expected_tool_call.get('args_validators', {}):
                        validators_match = all(
                            validator_check(k, validator)
                            for k, validator in expected_tool_call.get('args_validators', {}).items()
                        )
                except KeyError:
                    # Only possible in exact mode when key is missing
                    args_match = False
                    validators_match = False
                if args_match and validators_match:
                    cnt += 1
                    visited.add(idx)
                    break

    return cnt / len(expected_tool_calls) if not strict else float(cnt == len(expected_tool_calls))


def trace_to_str(agent_trace: List[ReadableSpan]) -> str:
    """Convert OTEL spans to a platform-style agent run history string.

    Creates a similar structure to LangChain message processing but using OTEL spans.
    Only processes tool spans (spans with 'tool.name' attribute).

    Args:
        agent_trace: List of ReadableSpan objects from the agent execution

    Returns:
        String representation of the agent run history in platform format
    """
    platform_history = []
    seen_tool_calls = set()

    for span in agent_trace:
        if span.attributes and (tool_name := span.attributes.get('tool.name')):
            # Get span timing information
            start_time = span.start_time
            end_time = span.end_time

            # Convert nanoseconds to datetime if needed
            if isinstance(start_time, int):
                start_timestamp = datetime.fromtimestamp(start_time / 1e9)
            else:
                start_timestamp = start_time

            if isinstance(end_time, int):
                end_timestamp = datetime.fromtimestamp(end_time / 1e9)
            else:
                end_timestamp = end_time

            timestamp_str = (
                start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if start_timestamp else ""
            )

            # Get tool call information
            tool_args = span.attributes.get('input.value', {})
            tool_result = span.attributes.get('output.value', '{}')
            # Attempt to extract only the content of the tool result if it is a string
            if isinstance(tool_result, str):
                try:
                    tool_result = json.loads(tool_result.replace("'", '"'))["content"]
                except (json.JSONDecodeError, KeyError):
                    tool_result = tool_result

            span_id = span.context.span_id if span.context else str(hash(f"{tool_name}_{timestamp_str}"))

            # De-duplicate tool calls based on span ID
            if span_id in seen_tool_calls:
                continue
            seen_tool_calls.add(span_id)

            # Add tool selection (equivalent to AIMessage with tool_calls)
            platform_history.append(f"[{timestamp_str}] LLM Response:")
            platform_history.append(f"  Agent Selected 1 Tool(s):")
            platform_history.append("")
            platform_history.append(f"  Tool: {tool_name}")
            platform_history.append(f"  Arguments: {str(tool_args)}")
            platform_history.append("")

            # Add tool response (equivalent to ToolMessage)
            end_timestamp_str = (
                end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if end_timestamp else timestamp_str
            )
            platform_history.append(f"[{end_timestamp_str}] Tool Call Response - {tool_name}:")
            platform_history.append(f"{str(tool_result).strip()}")
            platform_history.append("")

    return "\n".join(platform_history)


class AgentExecution(BaseAgentExecution):
    """Agent execution with additional fields."""
    simulation_instructions: str = ""
