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
uv run uipath run
```

### Agent Configuration

The library expects multiple files in order to run:

-   `agent.json`: agent definition file
-   `bindings.json`: agent's bindings definiton file
-   `uipath.json`: binding overwrite information

These files are generated automatically by Studio Web / Serverless Executor and can be obtained for local development by following these steps:

1. Open an agent project in Studio Web.
2. Start a Debug run (you can wait for the agent to finish or stop after a few seconds).
3. Run this script in the DevTools console:

    ```javascript
    await __agents.debug.getAgentBuilderDirectory();
    ```

    - if a warning with `.agent-builder folder not found` is displayed, run the agent in Debug one more time

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

### Commands

For normal local development, only the `run` command is relevant, as the `debug` command is not meant for local debug, it's intended for Studio Web debug. The `run` command can be used for debugging the python code IDE.

**Run**

The basic command used to start or resume an agent execution in production mode (source code [cli_run.py](src/uipath_agents/_cli/cli_run.py))

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

**Debug (StudioWeb)**

The command used for runs started from Studio Web. It loads the `agent.json` and `bindings.json` from the `.agent-builder` directory and copies them to the root directory and wires-up the debug hooks / channels. Should not be used for local development unless you're working on debug specific features.

```bash
#                   <entrypoint>  <args>
uv run uipath debug agent.json    '{}'
```

**Dev**

Currently work in progress. It's available for standard coded agents but not yet adapted for agents.

### Hints

-   Take a look at the configurations in `./examples`\_
-   You can run any agent from any sub directory of `uipath-agents-python` as long as you have the three required files present.
-   Remember to run `uv run uipath auth (--alpha)` if you get a 401 while executing the agent.
-   You can also download the solution package from Studio Web and get the files from the `.agent-builder` directory or from an already published package.
-   Do not confuse the `uipath debug` command with the `--debug` flag. The former is used for StudioWeb two-way communication for breakpoints, logs and so on and the latter waits for a debugger to be attached to the running process.

## Development Quickstart

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
