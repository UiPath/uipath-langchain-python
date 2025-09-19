
from collections.abc import Mapping
import json
from typing import Any, Dict, List, Set

from opentelemetry.sdk.trace import ReadableSpan


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
