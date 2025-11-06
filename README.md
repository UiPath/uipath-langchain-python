# UiPath LowCode Python

Lightweight library for building LowCode LangGraph Agents from JSON configuration files using the UiPath LangChain SDK.

## Overview

This library provides a simple interface to create LowCode LangGraph agents by reading configuration from `agent.json` files.
It handles:

- Loading and validating agent configuration
- Creating LLM instances with specified settings
- Building initial messages from templates
- Creating tools from resource definitions
- Delegating to `uipath-langchain` for the actual agent graph construction

The actual agent implementation (nodes, loops, execution logic) lives in the [`uipath-langchain`](https://github.com/UiPath/uipath-langchain-python) SDK.

## Prerequisites

- **Python 3.12+** - Required for modern type hints and features
- **uv** - Fast Python package manager ([install instructions](https://github.com/astral-sh/uv))

## Installation

```bash
# Clone the repository
cd uipath-lowcode-python

# Install dependencies (creates virtual environment automatically)
uv sync

# Install development dependencies
uv sync --group dev
```

## Project Structure

```
uipath-lowcode-python/
├── src/uipath_lowcode/
│   └── lowcode_agent_graph_builder/
│       ├── graph.py              # Main entry point - builds agent graph
│       ├── input_loader.py       # Loads agent.json and input data
│       ├── message_utils.py      # Message interpolation and building
│       ├── llm_utils.py          # LLM instance creation
│       ├── constants.py          # Configuration constants
│       ├── exceptions.py         # Custom exceptions
│       └── __init__.py
├── tests/
│   └── unit/
│       └── lowcode_agent_graph_builder/
│           ├── test_input_loader.py
│           ├── test_message_utils.py
│           └── ...
├── docs/                         # Additional documentation
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## Usage

### Basic Usage

```python
from uipath_lowcode.lowcode_agent_graph_builder import build_lowcode_agent_graph

# Build agent from agent.json in current directory with input data
graph = await build_lowcode_agent_graph(
    input_data={"topic": "Quarterly sales report"}
)
```

### Agent Configuration (`agent.json`)

The library expects an `agent.json` file in the current working directory with this structure:

```json
{
  "settings": {
    "model": "gpt-5-2025-08-07",
    "temperature": 0.9,
    "max_tokens": 16384
  },
  "messages": [
    {
      "role": "System",
      "content": "You are a helpful assistant."
    },
    {
      "role": "User",
      "content": "Research the given {{topic}} and output the given report."
    }
  ],
  "input_schema": {
    "type": "object",
    "properties": {
      "topic": {
          "type": "string"
      }
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "report": {
          "type": "string"
      }
    }
  },
  "resources": []
}
```

## Development

### Running the Agent

The project can be run via the UiPath CLI:

```bash
uv run uipath run lowcode-agent '{ "task": "say hello", "times": 2 }'
```

### Code Formatting & Linting

The project uses `ruff` for linting and formatting.

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix
```

### Type Checking

The project uses `mypy` for static type checking with strict configuration.

```bash
# Check entire codebase
uv run mypy src/uipath_lowcode

# Check specific file
uv run mypy src/uipath_lowcode/lowcode_agent_graph_builder/graph.py
```

### Running Tests

The project uses `pytest` for testing. All test commands should be run via `uv run` to ensure the correct virtual environment is used.

```bash
# Run all tests
uv run pytest tests/unit/ -v

# Run tests with coverage
uv run pytest tests/unit/ -v --cov=src/uipath_lowcode/lowcode_agent_graph_builder --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/lowcode_agent_graph_builder/test_input_loader.py -v

# Run specific test method
uv run pytest tests/unit/lowcode_agent_graph_builder/test_input_loader.py::TestLoadAgentConfiguration::test_valid_config -v
```

Coverage reports are generated in `htmlcov/index.html`.

### Local Development with Editable Dependencies

If you're developing against local versions of the UiPath SDK and UiPath LangChain SDK, you can configure editable installs:

**IMPORTANT**: **Do NOT commit this change**
1. **Add to `pyproject.toml`** under `[tool.uv.sources]`:
   ```toml
   [tool.uv.sources]
   uipath = { path = "../uipath-python", editable = true }
   uipath-langchain = { path = "../uipath-langchain-python", editable = true }
   ```

2. **Sync dependencies**:
   ```bash
   uv sync
   ```

### Development Workflow

1. **Install pre-commit hooks** (optional but recommended)

   Pre-commit automatically runs ruff linting and formatting before each commit:
   ```bash
   uv run pre-commit install
   ```

   # Run manually on all files
   ```bash
   pre-commit run --all-files
   ```

2. **Make changes** to the code

3. **Run tests** to ensure nothing breaks
   ```bash
   uv run pytest tests/unit/ -v
   ```

4. **Run type checks**
   ```bash
   uv run mypy .
   ```

5. **Format and lint**
   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   ```

## Architecture

This library follows a simple layered architecture:

1. **Configuration Layer** (`input_loader.py`)
   - Loads `agent.json` configuration
   - Validates input data against JSON schemas
   - Provides type-safe configuration objects

2. **Message Building Layer** (`message_utils.py`)
   - Interpolates message templates with input variables
   - Serializes complex data types for LLM consumption
   - Builds initial message list for agent

3. **LLM Setup Layer** (`llm_utils.py`)
   - Creates LLM instances with proper configuration
   - Handles model selection and parameters

4. **Graph Construction Layer** (`graph.py`)
   - Coordinates all components
   - Delegates to `uipath-langchain` for actual agent creation
   - Returns compiled LangGraph StateGraph

The actual agent execution logic (tool calling, routing, loops) is implemented in the `uipath-langchain` package.

## Related Projects

- [uipath-python](https://github.com/UiPath/uipath-python) - Core UiPath Python SDK
- [uipath-langchain](https://github.com/UiPath/uipath-langchain-python) - UiPath LangChain Integration SDK
