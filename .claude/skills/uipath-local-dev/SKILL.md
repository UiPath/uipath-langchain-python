---
name: uipath-local-dev
description: Run and test UiPath coded agents locally. Use when the user wants to run agents, execute evaluations, pack projects, or test UiPath Python agents locally. Triggers on commands like "run agent", "uipath run", "uipath eval", "test calculator", "local development", or "run evaluations".
allowed-tools: Bash, Read, Glob, Grep
---

# UiPath Local Development

Run and test UiPath coded agents locally using the `uipath` CLI commands.

## Quick Commands

### Run an Agent

Run an agent with input parameters (input is a positional argument, not an option):

```bash
cd examples/calculator && uv run uipath run agent.json '{"a": 5, "b": 3, "operator": "+"}'
```

### Run Evaluations

Run evaluations against an agent (auto-discovers eval sets in `evaluations/` folder):

```bash
cd examples/calculator && uv run uipath eval agent.json
```

### Authenticate

Before running agents, ensure you're authenticated with UiPath:

```bash
uv run uipath auth --alpha  # For alpha environment
uv run uipath auth --staging  # For staging environment
uv run uipath auth  # For production
```

### Pack a Project

Create a `.nupkg` package:

```bash
uv run uipath pack
```

### Publish to Orchestrator

Upload to Orchestrator:

```bash
uv run uipath publish -f "<folder-name>"
```

## Available Examples

The repository contains several example agents in the `examples/` directory:

- **calculator**: A simple calculator agent with evaluations
- **basic**: Basic agent example
- **basic_with_ootb_guardrails**: Agent with out-of-the-box guardrails
- **debug**: Debug agent example

## Environment Setup

Before running agents, ensure dependencies are synced:

```bash
uv sync
```

## Local Testing with Custom Repositories (Hacked-Coded Setup)

For testing local changes across multiple UiPath Python repositories, you need to set up a "hacked-coded" environment.

### Requirements for Local Testing

To test local changes to core UiPath libraries, you need these additional repositories:

1. **uipath-python** - Core UiPath Python SDK
2. **uipath-langchain-python** - LangChain integration
3. **uipath-agents-python** - This repository (agents runtime)

### Setup Instructions

1. **Create a parent directory** and clone all repos:

```bash
mkdir hacked-coded && cd hacked-coded
git clone <uipath-agents-python-repo>
git clone <uipath-langchain-python-repo>
git clone <uipath-python-repo>
```

2. **Configure pyproject.toml** with editable dependencies:

```toml
[tool.uv.sources]
uipath = { path = "./uipath-python", editable = true }
uipath-langchain = { path = "./uipath-langchain-python", editable = true }
uipath-agents = { path = "./uipath-agents-python", editable = true }
```

3. **Apply the required patch** to `uipath-python` for nested repo packing:

```bash
cd uipath-python
git apply ../nested-repos.patch
```

4. **Sync all dependencies**:

```bash
cd uipath-python && uv sync && cd ..
cd uipath-langchain-python && uv sync && cd ..
cd uipath-agents-python && uv sync && cd ..
uv sync
```

### Patch Content

The patch modifies `src/uipath/_cli/_utils/_project_files.py` to handle nested repositories during packing. See the Confluence documentation for the full patch.

## Workflow Reference

| Command | Description |
| --- | --- |
| `uv run uipath run <agent.json>` | Run an agent locally |
| `uv run uipath run <agent.json> '{...}'` | Run with specific input (JSON positional arg) |
| `uv run uipath eval <agent.json>` | Run evaluations |
| `uv run uipath pack` | Create package |
| `uv run uipath publish -f <folder>` | Publish to Orchestrator |

## Troubleshooting

- **Missing dependencies**: Run `uv sync` to install dependencies
- **Authentication issues**: Check `.uipath/.auth.json` for credentials
- **Model errors**: Verify `.env` file has correct API keys configured
