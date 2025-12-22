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

-   **uv** - Package and Python Manager ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))
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
uv run uipath run agent.json '{}'
```

### Agent Configuration

The library expects multiple files in order to run:

-   `agent.json`: agent definition file (also used for Agents runtime detection)
-   `bindings.json`: agent's bindings definiton file
-   `uipath.json`: binding overwrite information

These files must be present at the **top level** of your working directory. The UiPath CLI automatically detects which runtime to use based on the presence of `agent.json` - if found, it loads the `AgentsRuntimeFactory` from this package to handle agent execution.

#### Getting the Configuration Files

These files are generated automatically by Studio Web / Serverless Executor. For local development, you can obtain them by following these steps:

1. Open an agent project in Studio Web.
2. Start a Debug run (you can wait for the agent to finish or stop after a few seconds).
3. Run this script in the DevTools console:

    ```javascript
    await __agents.debug.getAgentBuilderDirectory();
    ```

    - if a warning with `.agent-builder folder not found` is displayed, run the agent in Debug one more time

4. Copy the `agent.json` and `bindings.json` from the output to your local project directory.
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

**Note on `.agent-builder` directory**: When running agents locally, the runtime automatically checks for a `.agent-builder/` directory. If it exists, any files inside will be copied to the top level, overriding existing files. This is used by Studio Web, but you can also leverage it for local testing by placing updated configurations in `.agent-builder/` without modifying your main files.

### Commands

For normal local development, only the `run` command is relevant, as the `debug` command is not meant for local debug, it's intended for Studio Web debug. The `run` command can be used for debugging the python code IDE.

**Run**

The basic command used to start or resume an agent execution in production mode (source code [cli_run.py](src/uipath_agents/_cli/cli_run.py)). This command will automatically prepare agent files from `.agent-builder/` if that directory exists before executing.

```bash
# Start the agent.json from the current directory with inline args
#                 <entrypoint>   <args>
uv run uipath run agent.json     '{}'

# Start the agent.json from the current directory with file-based args
#                            <args from file>
uv run uipath run agent.json -f input.json

# Start the agent.json from the current directory with no args (short command)
uv run uipath run

# Resume an agent's execution after an interruption (process, hitl and so on)
uv run uipath run --resume
```

-   **Entrypoint** is currently unused for Agents, it can be be any string or removed entirely.
-   Files from `.agent-builder/` are automatically copied to the top level if that directory exists

**Debug (StudioWeb)**

The command used for runs started from Studio Web. Like the `run` command, it automatically prepares files from `.agent-builder/` if that directory exists, then wires-up the debug hooks and communication channels with Studio Web. Should not be used for local development unless you're working on debug specific features.

```bash
#                   <entrypoint>  <args>
uv run uipath debug agent.json    '{}'
```

-   Files are loaded from `.agent-builder/` and copied to the top level before execution
-   Establishes bidirectional communication with Studio Web for breakpoints, logs, and state inspection

**Eval**

The command used to run evaluation sets against your agent (source code from the `uipath` CLI package). Evaluations help you test agent behavior against expected outcomes using various evaluators like exact matching, similarity scoring, and trajectory analysis.

```bash
# Run evaluations from the current directory (auto-discovers agent and eval-set)
uv run uipath eval

# Run a specific eval-set file
uv run uipath eval agent.json evaluations/eval-sets/default.json

# Run with multiple parallel workers for faster execution
uv run uipath eval --workers 4

# Run specific evaluations by ID
uv run uipath eval --eval-ids "['test-basic', 'test-advanced']"

# Run without reporting results to Studio Web
uv run uipath eval --no-report

# Save evaluation results to a file
uv run uipath eval --output-file results.json
```

**Evaluation Structure**

The evaluation framework expects an `evaluations/` directory with:
-   `eval-sets/`: JSON files defining test cases and evaluation criteria
-   `evaluators/`: JSON files defining how to evaluate agent outputs (exact match, similarity, contains, etc.)

See `examples/basic/evaluations/` for a complete example of how to structure your evaluation files.

**Dev**

Currently work in progress. It's available for standard coded agents but not yet adapted for agents.

### Hints

-   Take a look at the configurations in `./examples`
-   You can run any agent from any sub directory of `uipath-agents-python` as long as you have the three required files present at the top level of that directory.
-   Remember to run `uv run uipath auth (--alpha)` if you get a 401 while executing the agent.
-   You can also download the solution package from Studio Web and get the files from the `.agent-builder` directory or from an already published package.
-   Do not confuse the `uipath debug` command with the `--debug` flag. The former is used for StudioWeb two-way communication for breakpoints, logs and so on and the latter waits for a debugger to be attached to the running process.

## Development Quickstart

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
