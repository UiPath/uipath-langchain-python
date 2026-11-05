# DeepAgent workspace files

A [DeepAgents](https://github.com/langchain-ai/deepagents) agent that writes a report to a real file, suspends for a human review, and revises the same file after it resumes. In an Orchestrator job the file survives the suspend, even when the job resumes on another machine.

## Overview

`UiPathWorkspaceBackend` gives the agent's file tools (`write_file`, `read_file`, `edit_file`, `ls`, `glob`, `grep`) a folder on disk that belongs to the current run. When the job suspends, the UiPath CLI uploads that folder as job attachments. When the job resumes, it downloads them into a new folder before the agent continues.

Use it when the agent's files need to exist on disk. In this sample, the `request_review` tool opens `report.md` with plain Python, from `backend.cwd`, and sends its text to the reviewer. DeepAgents' default `StateBackend` keeps files inside the graph state instead, where your own tools can't open them.

## Requirements

- Python 3.11+
- A UiPath Automation Cloud tenant. The model runs through the UiPath LLM Gateway, so no model API key is needed.

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
uipath auth
```

## Usage

Run it as a job, so the files are kept:

```bash
uipath pack
uipath publish
```

Start the process in Orchestrator with this input:

```json
{"messages": [{"type": "human", "content": "Write a short report on the benefits of unit tests"}]}
```

The job writes `/report.md`, calls `request_review` and suspends. Resume it with the reviewer's feedback, for example `{"feedback": "Add a section on test speed"}`. The agent then edits the same `/report.md` and replies with the final report.

You can also run it locally:

```bash
uipath run agent '{"messages": [{"type": "human", "content": "Write a short report on the benefits of unit tests"}]}'
uipath run agent '{"feedback": "Add a section on test speed"}' --resume
```

Outside a job the backend uses a temporary folder and logs a warning naming it. Each local run gets a new folder, so after `--resume` the draft is gone: `edit_file` reports `/report.md` as not found, and the agent writes a new report from scratch. Only jobs keep the workspace, because only jobs resume in a new process.

## How it works

1. The UiPath CLI sees that the LangGraph factory opts in to a managed workspace, and wraps each job run with it.
2. Before the run, the CLI restores the files recorded for this job into a new folder and makes it the run's workspace.
3. `UiPathWorkspaceBackend` looks up that folder on every file operation, so a graph built once at import time always uses the current run's folder.
4. When the run suspends, the CLI uploads the folder's changed files as job attachments and records them for the next run.

Your own tools can open the same folder through `backend.cwd`, or with `uipath.runtime.get_workspace_path()`, which raises outside a managed run.
