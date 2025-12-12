"""Refactoring guide tool that maps code issues to refactoring prompts."""

from typing import Any, Dict


def get_refactoring_guide(issue_type: str) -> Dict[str, Any]:
    """Get guidance on which refactoring prompt to use for a specific issue.

    This is the bridge tool that maps detected code issues to appropriate refactoring prompts.

    Args:
        issue_type: Type of code issue (e.g., "long_function", "deep_nesting", "poor_naming")

    Returns:
        Dictionary with recommended prompt name and details
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

    return issue_to_prompt[issue_type]
