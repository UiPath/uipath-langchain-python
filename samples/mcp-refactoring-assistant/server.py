from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import Message

from prompts import extract_method, simplify_conditional, remove_duplication, improve_naming
from tools import code_analysis, refactoring_guide

mcp = FastMCP("code-refactoring-assistant")


# Register Code Analysis Tools
@mcp.tool()
def analyze_code_complexity(code: str) -> Dict[str, Any]:
    """Analyze Python code complexity using heuristics.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary with complexity metrics and detected issues
    """
    return code_analysis.analyze_code_complexity(code)


@mcp.tool()
def detect_code_smells(code: str) -> Dict[str, Any]:
    """Detect common code smells in Python code.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary with detected code smells and their locations
    """
    return code_analysis.detect_code_smells(code)


@mcp.tool()
def get_refactoring_guide(issue_type: str) -> Dict[str, Any]:
    """Get guidance on which refactoring prompt to use for a specific issue.

    This is the bridge tool that maps detected code issues to appropriate refactoring prompts.

    Args:
        issue_type: Type of code issue (e.g., "long_function", "deep_nesting", "poor_naming")

    Returns:
        Dictionary with recommended prompt name and details
    """
    return refactoring_guide.get_refactoring_guide(issue_type)


# Register Refactoring Prompts
@mcp.prompt()
def extract_method_prompt(code: str, target_lines: str = "auto") -> List[Message]:
    """Guide for extracting a method from complex code.

    Args:
        code: Full Python function/method code to refactor
        target_lines: Lines to extract (e.g., "10-25" or "auto" for suggestions)

    Returns:
        Structured guidance messages for Extract Method refactoring
    """
    return extract_method(code, target_lines)


@mcp.prompt()
def simplify_conditional_prompt(code: str, pattern: str = "guard_clause") -> List[Message]:
    """Guide for simplifying complex conditional logic.

    Args:
        code: Python code with complex conditionals
        pattern: Refactoring pattern (guard_clause, early_return, extract_condition)

    Returns:
        Structured guidance for simplifying conditionals
    """
    return simplify_conditional(code, pattern)


@mcp.prompt()
def remove_duplication_prompt(code: str, duplicate_blocks: str) -> List[Message]:
    """Guide for removing code duplication.

    Args:
        code: Python code with duplication
        duplicate_blocks: Description of what's duplicated

    Returns:
        Structured guidance for removing duplication
    """
    return remove_duplication(code, duplicate_blocks)


@mcp.prompt()
def improve_naming_prompt(code: str, symbols: str, context: str = "") -> List[Message]:
    """Guide for improving variable and function names.

    Args:
        code: Python code with poor names
        symbols: Symbols to rename (comma-separated)
        context: Optional description of what the code does

    Returns:
        Structured guidance for improving names
    """
    return improve_naming(code, symbols, context)


# Run the server when the script is executed
if __name__ == "__main__":
    mcp.run()
