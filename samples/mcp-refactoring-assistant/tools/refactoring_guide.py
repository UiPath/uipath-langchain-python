"""Refactoring guide tool that maps code issues to refactoring prompts."""


def get_refactoring_guide(
    issue_type: str,
    code: str = "",
    complexity_info: dict | None = None,
    smells_info: dict | None = None
) -> dict:
    """Get guidance on which refactoring prompt to use for a specific issue.


    Args:
        issue_type: Type of code issue (e.g., "long_function", "deep_nesting", "poor_naming")
        code: The code to refactor (required for constructing prompt arguments)
        complexity_info: Optional complexity analysis results
        smells_info: Optional code smells detection results

    Returns:
        Dictionary with prompt_name and arguments for te prompt.
        Format: {"prompt_name": "...", "arguments": {...}}
    """
    # Mapping from issue types to MCP prompts
    issue_to_prompt = {
        "long_function": {
            "prompt_name": "extract_method_prompt",
            "required_args": ["code", "target_lines"],
            "description": "Extract a method to reduce function length and complexity",
            "priority": "high",
            "issue_explanation": "Long functions are hard to understand, test, and maintain"
        },
        "high_complexity": {
            "prompt_name": "extract_method_prompt",
            "required_args": ["code", "target_lines"],
            "description": "Break down complex logic into smaller functions",
            "priority": "high",
            "issue_explanation": "High complexity makes code difficult to reason about"
        },
        "deep_nesting": {
            "prompt_name": "simplify_conditional_prompt",
            "required_args": ["code", "pattern"],
            "description": "Simplify nested conditionals using guard clauses or early returns",
            "priority": "high",
            "issue_explanation": "Deep nesting reduces readability and increases cognitive load"
        },
        "multiple_returns": {
            "prompt_name": "simplify_conditional_prompt",
            "required_args": ["code", "pattern"],
            "description": "Organize return statements for better flow control",
            "priority": "medium",
            "issue_explanation": "Multiple returns can make control flow confusing"
        },
        "poor_naming": {
            "prompt_name": "improve_naming_prompt",
            "required_args": ["code", "symbols", "context"],
            "description": "Improve variable and function names following PEP 8",
            "priority": "medium",
            "issue_explanation": "Poor names reduce code clarity and maintainability"
        },
        "long_parameter_list": {
            "prompt_name": "extract_method_prompt",
            "required_args": ["code", "target_lines"],
            "description": "Consider using parameter objects or breaking down the function",
            "priority": "medium",
            "issue_explanation": "Long parameter lists are hard to understand and error-prone"
        },
        "duplicate_code": {
            "prompt_name": "remove_duplication_prompt",
            "required_args": ["code", "duplicate_blocks"],
            "description": "Extract duplicated code into reusable functions",
            "priority": "high",
            "issue_explanation": "Duplication leads to maintenance burden and inconsistency"
        },
        "bare_except": {
            "prompt_name": "simplify_conditional_prompt",
            "required_args": ["code", "pattern"],
            "description": "Specify exception types for better error handling",
            "priority": "high",
            "issue_explanation": "Bare except clauses can hide bugs and make debugging difficult"
        },
        "long_conditional_chain": {
            "prompt_name": "simplify_conditional_prompt",
            "required_args": ["code", "pattern"],
            "description": "Refactor long if/elif chains using dictionaries or polymorphism",
            "priority": "medium",
            "issue_explanation": "Long conditional chains are hard to maintain and extend"
        }
    }

    if issue_type not in issue_to_prompt:
        return {
            "error": f"Unknown issue type: {issue_type}",
            "available_types": list(issue_to_prompt.keys()),
            "suggestion": "Use analyze_code_complexity or detect_code_smells to identify issues"
        }

    guide = issue_to_prompt[issue_type]
    prompt_name = guide["prompt_name"]

    # Construct arguments dict based on prompt requirements
    if prompt_name == "extract_method_prompt":
        return {
            "prompt_name": prompt_name,
            "arguments": {
                "code": code,
                "target_lines": "auto"
            },
            "description": guide["description"],
            "priority": guide["priority"]
        }

    if prompt_name == "simplify_conditional_prompt":
        # Determine pattern based on issue type
        pattern = "guard_clause" if issue_type == "deep_nesting" else "early_return"
        return {
            "prompt_name": prompt_name,
            "arguments": {
                "code": code,
                "pattern": pattern
            },
            "description": guide["description"],
            "priority": guide["priority"]
        }

    if prompt_name == "remove_duplication_prompt":
        # Extract duplicate info from smells_info if available
        duplicate_blocks = "auto-detected duplicates"
        if smells_info and "duplicate_code" in smells_info.get("smells", []):
            duplicate_blocks = "See code analysis for duplicate blocks"
        return {
            "prompt_name": prompt_name,
            "arguments": {
                "code": code,
                "duplicate_blocks": duplicate_blocks
            },
            "description": guide["description"],
            "priority": guide["priority"]
        }

    if prompt_name == "improve_naming_prompt":
        # Extract poor naming symbols from smells_info
        symbols = "x,y,z"  # Default, agent can override
        if smells_info:
            poor_names = [s.get("symbol", "") for s in smells_info.get("smells", [])
                         if s.get("type") == "poor_naming"]
            if poor_names:
                symbols = ",".join(poor_names)
        return {
            "prompt_name": prompt_name,
            "arguments": {
                "code": code,
                "symbols": symbols,
                "context": ""  # Optional, agent can provide
            },
            "description": guide["description"],
            "priority": guide["priority"]
        }

    # Fallback: just provide code
    return {
        "prompt_name": prompt_name,
        "arguments": {"code": code},
        "description": guide["description"],
        "priority": guide["priority"]
    }

