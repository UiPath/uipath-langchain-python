"""Code analysis tools for detecting complexity and code smells."""

import re


def analyze_code_complexity(code: str) -> dict:
    """Analyze Python code complexity using heuristics.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary with complexity metrics and detected issues
    """
    lines = code.split('\n')
    line_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])

    # Count complexity indicators
    complexity_score = 0
    complexity_score += len(re.findall(r'\bif\b', code))
    complexity_score += len(re.findall(r'\belif\b', code))
    complexity_score += len(re.findall(r'\bfor\b', code))
    complexity_score += len(re.findall(r'\bwhile\b', code))
    complexity_score += len(re.findall(r'\btry\b', code))
    complexity_score += len(re.findall(r'\bexcept\b', code))
    complexity_score += len(re.findall(r'\bwith\b', code))
    complexity_score += len(re.findall(r'\band\b', code))
    complexity_score += len(re.findall(r'\bor\b', code))

    # Count functions
    function_count = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))

    # Detect nesting depth
    max_nesting = 0
    for line in lines:
        if line.strip():
            # Count leading spaces divided by 4 (assuming 4-space indentation)
            indent = (len(line) - len(line.lstrip())) // 4
            max_nesting = max(max_nesting, indent)

    # Detect issues
    issues = []
    if line_count > 50:
        issues.append("long_function")
    if complexity_score > 10:
        issues.append("high_complexity")
    if max_nesting > 3:
        issues.append("deep_nesting")

    # Count multiple returns
    return_count = len(re.findall(r'^\s*return\b', code, re.MULTILINE))
    if return_count > 3:
        issues.append("multiple_returns")

    return {
        "line_count": line_count,
        "complexity_score": complexity_score,
        "function_count": function_count,
        "max_nesting_depth": max_nesting,
        "return_statements": return_count,
        "issues": issues,
        "summary": f"Complexity: {complexity_score}, Lines: {line_count}, Max nesting: {max_nesting}"
    }


def detect_code_smells(code: str) -> dict:
    """Detect common code smells in Python code.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary with detected code smells and their locations
    """
    lines = code.split('\n')
    smells = []

    # Check for long functions
    function_starts = []
    for i, line in enumerate(lines, 1):
        if re.match(r'^\s*def\s+\w+', line):
            function_starts.append(i)

    if len(function_starts) > 0:
        for i, start in enumerate(function_starts):
            end = function_starts[i + 1] if i + 1 < len(function_starts) else len(lines)
            length = end - start
            if length > 50:
                smells.append({
                    "type": "long_function",
                    "line": start,
                    "severity": "high",
                    "description": f"Function starting at line {start} is {length} lines long (>50)"
                })

    # Check for deep nesting
    for i, line in enumerate(lines, 1):
        if line.strip():
            indent_level = (len(line) - len(line.lstrip())) // 4
            if indent_level > 3:
                smells.append({
                    "type": "deep_nesting",
                    "line": i,
                    "severity": "medium",
                    "description": f"Line {i} has nesting depth of {indent_level} (>3)"
                })

    # Check for long parameter lists
    long_params = re.finditer(r'def\s+\w+\s*\((.*?)\):', code, re.DOTALL)
    for match in long_params:
        params = [p.strip() for p in match.group(1).split(',') if p.strip() and p.strip() != 'self']
        if len(params) > 4:
            line_num = code[:match.start()].count('\n') + 1
            smells.append({
                "type": "long_parameter_list",
                "line": line_num,
                "severity": "medium",
                "description": f"Function has {len(params)} parameters (>4)"
            })

    # Check for poor naming (single letter variables, excluding common loop vars)
    poor_names = re.finditer(r'\b([a-z])\s*=', code)
    for match in poor_names:
        var_name = match.group(1)
        if var_name not in ['i', 'j', 'k', 'x', 'y', 'z']:  # Allow common math/loop vars
            line_num = code[:match.start()].count('\n') + 1
            smells.append({
                "type": "poor_naming",
                "line": line_num,
                "severity": "low",
                "description": f"Single letter variable '{var_name}' at line {line_num}"
            })

    # Check for missing docstrings
    functions_without_docstring = []
    for i, line in enumerate(lines):
        if re.match(r'^\s*def\s+\w+', line):
            # Check if next non-empty line is a docstring
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip():
                    if not lines[j].strip().startswith('"""') and not lines[j].strip().startswith("'''"):
                        functions_without_docstring.append(i + 1)
                    break

    for line_num in functions_without_docstring:
        smells.append({
            "type": "missing_docstring",
            "line": line_num,
            "severity": "low",
            "description": f"Function at line {line_num} missing docstring"
        })

    # Check for bare except clauses
    bare_excepts = re.finditer(r'except\s*:', code)
    for match in bare_excepts:
        line_num = code[:match.start()].count('\n') + 1
        smells.append({
            "type": "bare_except",
            "line": line_num,
            "severity": "high",
            "description": f"Bare except clause at line {line_num} catches all exceptions"
        })

    # Check for long if/elif chains
    elif_chains = re.finditer(r'(if\b.*?:.*?)(\n\s*elif\b.*?:.*?){3,}', code, re.DOTALL)
    for match in elif_chains:
        line_num = code[:match.start()].count('\n') + 1
        chain_length = match.group(0).count('elif') + 1
        smells.append({
            "type": "long_conditional_chain",
            "line": line_num,
            "severity": "medium",
            "description": f"Long if/elif chain with {chain_length} conditions at line {line_num}"
        })

    return {
        "total_smells": len(smells),
        "smells": smells,
        "summary": f"Found {len(smells)} code smells"
    }
