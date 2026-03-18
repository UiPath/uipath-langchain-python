# Web Researcher — Deep Agent Example

A research agent that produces comprehensive reports by combining planning, web search, file-based note-taking, and context summarization — powered by the UiPath deep agent harness.

## Overview

Unlike a simple ReAct loop, this agent uses several inbuilt tools/middleware layers to handle long-running research tasks:

- **TodoListMiddleware** — provides `write_todos` and `read_todos` tools so the agent can plan work upfront, track progress, and stay organized through long-running tasks
- **Subagents** — delegates independent research tasks to focused, parallel agents with isolated contexts; each subagent receives detailed instructions and returns concise findings to the orchestrator
- **FileSystemBackend** — enables agents to persist research notes and drafts to disk using `read_file`, `write_file`, and `edit_file` tools, creating a durable scratch space for organizing findings
- **SummarizationMiddleware** — auto-compresses conversation history when approaching token limits, archiving old messages to persistent storage so the agent can handle extended research workflows

The agent also has access to **Web Search** (UiPath Integration Service) to perform web searches.

## Requirements

- Python 3.11+
- UiPath authentication (`uv run uipath auth --alpha`)
- A UiPath connection for Web Search (Google Custom Search)

## Usage

From the repository root:

```bash
uv run uipath run examples/web_researcher/agent.json '{"prompt": "Analyze the current state of open-source AI vs closed-source AI"}'
```

Or use the provided `input.json`:

```bash
uv run uipath run agent.json -f input.json  --output-file deep_result.json
```

## How It Works

1. **PLAN** — decomposes the research prompt into focused sub-questions
2. **RESEARCH** — investigates each sub-question using varied web search queries
3. **REVIEW** — evaluates coverage, identifies gaps and contradictions
4. **GAP FILL** — runs targeted follow-up searches for missing information
5. **ASSEMBLE** — synthesizes findings into a structured report (max 10,000 tokens)


## Configuration

Key settings in `agent.json`:

| Setting | Value | Description |
|---------|-------|-------------|
| `model` | `anthropic.claude-sonnet-4-5-20250929-v1:0` | LLM model |
| `maxTokens` | `32000` | Max output tokens per LLM call |
| `temperature` | `0` | Deterministic output |
| `maxIterations` | `100` | Max agent loop iterations |
| `metadata.deepAgent` | `true` | Enables the deep agent harness |

