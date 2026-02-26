# Code Refactoring Assistant with MCP

A LangGraph agent that demonstrates how to use **MCP (Model Context Protocol)** for code analysis and refactoring guidance.

## What It Does

This agent analyzes Python code, detects issues (complexity, code smells, deep nesting), and provides tailored refactoring guidance using MCP prompts.

Shows how to use `client.get_prompt()` to fetch prompt templates dynamically from an MCP server.


## MCP Components

### Server (`server.py`)
Exposes:
- **Tools**: Code analysis functions
  - `analyze_code_complexity` - Detects complexity metrics
  - `detect_code_smells` - Finds common issues
  - `get_refactoring_guide` - Returns which prompt to use

- **Prompts**: Refactoring templates
  - `extract_method_prompt` - For long functions
  - `simplify_conditional_prompt` - For nested conditions
  - `remove_duplication_prompt` - For duplicate code
  - `improve_naming_prompt` - For poor variable names

### Client (`graph.py`)
- Agent uses tools to analyze code
- Agent determines which refactoring approach to use
- **`client.get_prompt()`** fetches the appropriate template from MCP server
- LLM generates final refactoring guidance

## How to Run

1. **Install dependencies**:
```bash
uv sync
```

2. **Initialize** (if needed):
```bash
uipath init
```

3. **Run the agent**:
```bash
uipath run agent --input-file input.json
```

## Example Input

```json
{
  "code": "def process_data(x, y, z):\n    if x:\n        if y:\n            if z:\n                return x + y + z"
}
```

## Example Output

The agent will:
1. Detect the issue (deep nesting)
2. Fetch the `simplify_conditional_prompt` via MCP
3. Generate refactoring guidance using guard clauses

