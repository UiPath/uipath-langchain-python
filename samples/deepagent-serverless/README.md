# DeepAgent Serverless Sandbox

A DeepAgent that delegates code execution to a serverless sandbox pod via `interrupt(InvokeProcess(...))`.

## What It Demonstrates

- **Serverless code execution**: The agent writes Python scripts and executes them in UiPath Orchestrator, without needing a local runtime.
- **Operation batching**: Writes are buffered and flushed together with execute commands in a single `InvokeProcess` call..
- **Subagent delegation**: A `research_specialist` subagent uses Tavily web search to gather information before the main agent writes code.
- **`SandboxBackendProtocol`**: Extends the deepagents `BackendProtocol` with an `execute` tool, giving the LLM the ability to run shell commands.

## Architecture

```
User prompt
    │
    ▼
┌─────────────────────────────────────────┐
│  agent (DeepAgent + LLM)                │
│                                         │
│  Tools: write_file, execute, read, ...  │
│  Subagents: research_specialist         │
│                                         │
│  ServerlessBackend buffers writes,      │
│  then flushes them + execute as a       │
│  single interrupt(InvokeProcess(...))   │
└───────────────┬─────────────────────────┘
                │ InvokeProcess
                ▼
┌─────────────────────────────────────────┐
│  sandbox (non-LLM StateGraph)           │
│  Runs in UiPath Orchestrator            │
│                                         │
│  Receives batched operations:           │
│  [write, write, ..., execute]           │
│  Executes them sequentially via         │
│  FilesystemBackend + subprocess         │
└─────────────────────────────────────────┘
```

## Entry Points

Defined in `langgraph.json`:

| Entry Point | Graph | Description |
|-------------|-------|-------------|
| `agent` | `graph.py:deep_agent` | LLM-powered DeepAgent with sandbox backend |
| `sandbox` | `sandbox.py:graph` | Non-LLM graph that executes batched file/shell operations |

## Requirements

- Python 3.11+
- Tavily API key (for web search)

## Setup

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Run `uv run uipath auth` to authenticate to UiPath.


Add your Tavily API key in `.env`.
```bash
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

```bash
uv run uipath run agent --file input.json
```

## Example Prompt

```
Research the best practices for benchmarking sorting algorithms in Python.
Then write a script that compares bubble sort, merge sort, and Python's
built-in timsort on random lists of 10000 elements, and execute it.
```

This exercises the full flow: research (Tavily) → write code (buffered) → execute → return results.
