# UiPath Agents Python

Lightweight library for running `UiPath - Agent Builder` using the UiPath LangChain SDK.

## Overview

This library provides a simple interface to create LangGraph agents by reading configuration from `agent.json` files.
It handles the high level operations:

-   Loading and validating agent configuration
-   Creating LLM instances with specified settings
-   Building initial messages from templates
-   Creating tools from resource definitions
-   Delegating to `uipath-langchain` for the actual agent graph construction and runtime

The actual agent implementation (nodes, loops, execution logic) lives in the [`uipath-langchain`](https://github.com/UiPath/uipath-langchain-python) SDK.

## Quick Start

### Prerequisites

-   **uv** - Package and Python Manager ([install instructions](https://github.com/astral-sh/uv))
    -   Python is automatically managed by `uv`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd uipath-agents-python

# Install dependencies (downloads Python and creates virtual environment automatically)
uv sync
```

### Choose your way of running python

-   Prefix all commands with `uv run` (e.g. `uv run python ...`, `uv run uipath ...` and so on)
-   Activate the virtual environment by running this command in the cloned directory
    -   Windows: `.\.venv\bin\activate.ps1`
    -   Linux : `source ./.venv/bin/activate`

_This document will use the first option, with `uv run` for any commands._

### Verify Installation

```bash
# Verify installation
uv run uipath --help
```

## Usage

### Basic Example

```bash
cd examples/basic

uv run uipath auth --alpha
uv run uipath run
```

### Agent Configuration

The library expects multiple files in order to run:

-   `agent.json`: agent definition file
-   `bindings.json`: agent's bindings definiton file
-   `uipath.json`: binding overwrite information

These files are generated automatically by Studio Web / Serverless Executor and can be obtained for local development by following these steps:

1. Open an agent project in Studio Web.
2. Start a Debug run (you can wait for it to finish or stop it after a few seconds).
3. Run this script in the DevTools console:

    ```javascript
    await __agents.debug.getAgentBuilderDirectory();
    ```

    - if a warning with `.agent-builder folder not found` is displayed, run the agent in Debug again

4. Copy the `agent.json` and `bindings.json` from the output.
5. Create a `uipath.json` file using the following contents:

    ```json
    {
        "runtime": {
            "internalArguments": {
                "resourceOverwrites": {}
            }
        }
    }
    ```

### Hints

-   Take a look at the configurations in `./examples`\_
-   You can run any agent from any sub directory of `uipath-agents-uipath` as long as you have the three required files present.
-   Remember to run `uv run uipath auth (--alpha)` if you get a 401 while executing the agent.
-   You can also download the solution package from Studio Web and get the files from the `.agent-builder` directory or from an already published package.

## Development Quickstart

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

### Code Quality Tools

#### Formatting & Linting (Ruff)

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Link and auto-fix issues
uv run ruff check . --fix
```

#### Type Checking (mypy)

```bash
# Check entire codebase
uv run mypy .

# Check specific file
uv run mypy src/uipath_agents/agent_graph_builder/graph.py
```

### Running Tests

TBA

## Local Development with Editable Dependencies

Most of the times when working on this repo, you'll also need to do some changes in `uipath-langchain-python` and/or `uipath-python`. In order to work with all 3 repositories, you'll need to configure them as `editable` dependencies.

### Setup

1. **Clone the dependency repositories** alongside this one:

    ```bash
    cd ..
    git clone [<uipath-langchain-python-url>](https://github.com/UiPath/uipath-langchain-python/)
    git clone [<uipath-python-url>](https://github.com/UiPath/uipath-python)

    # Your directory structure should look like:
    # .
    # ├── uipath-python/
    # ├── uipath-langchain-python/
    # └── uipath-agents-python/
    ```

2. **Update `pyproject.toml`** in `uipath-agents.python` and add the following section:

    ```toml
    [tool.uv.sources]
    uipath = { path = "../uipath-python", editable = true }
    uipath-langchain = { path = "../uipath-langchain-python", editable = true }
    ```

3. **Sync dependencies** for `uipath-agents-python`:

    ```bash
    uv sync
    ```

4. **Update `pyproject.toml`** in `uipath-langchain-python` and add the following section:

    ```toml
    [tool.uv.sources]
    uipath = { path = "../uipath-python", editable = true }
    ```

5. **Sync dependencies** for `uipath-langchain-python`:

    ```bash
    uv sync
    ```

6. **Go back to the agents repo**:

    ```bash
    cd ../uipath-agents-python
    ```

**IMPORTANT**: Do NOT commit the `editable` related changes to version control. This is for local development only. Using editable dependencies also updates the `uv.lock` files. If you need to update other dependencies in `pyproject.toml` files that also have `editable` dependencies, make sure to first revert the changes and do a clean `uv sync` to update the lockfile without the editable information and commit the changes before making them editable again.

## Related Projects

-   [uipath-python](https://github.com/UiPath/uipath-python) - Core UiPath Python SDK
-   [uipath-langchain-python](https://github.com/UiPath/uipath-langchain-python) - UiPath LangChain Integration SDK
