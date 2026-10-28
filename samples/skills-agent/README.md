# Skills Agent

A LangGraph agent that answers accounts-payable questions with skills from two bound skill bundles. It shows how a coded agent declares its bundles with `UiPathSkills`, resolves them once in a graph factory, lists the skills in its system prompt and reads a skill's full `SKILL.md` on demand through the `load_skill` tool.

See [Skills](../../docs/skills.md) for the LangChain patterns and the SDK's [Skills](https://uipath.github.io/uipath-python/core/skills/) page for how bundles are resolved.

> **Preview:** the Orchestrator Skills API is not live yet. This sample serves its own `skills/` folder through the Skills API client (`UIPATH_SKILLS_SOURCE=mock`), so no bundle has to exist in Orchestrator. The chat model still runs through the UiPath LLM gateway, so you need a tenant.

## What it contains

| File | Purpose |
|------|---------|
| `main.py` | The agent: `SKILLS = UiPathSkills([...])` at module level, and `make_graph()`, the graph factory `langgraph.json` points to. |
| `skills/` | The bundles: `finance` v3 (`invoice-triage`, `vendor-lookup`) and `ops` v1 (`escalation-policy`, `vendor-lookup`). |
| `bindings.json` | One `skillBundle` binding per bundle reference: `finance` pinned to version 3, `ops` on latest. |
| `.env.example` | The skills source variables. |
| `input.json` | A task that should make the model load `ops/escalation-policy`. |
| `resource-overwrites.example.json` | A per-environment override of the `ops` binding, to try locally. |

The graph is an `agent` node and a `tools` node (`ToolNode` with `load_skill`) in a loop, then a `finish` node that returns:

```json
{"answer": "...", "skills_used": ["ops/escalation-policy"]}
```

`skills_used` lists the skills `load_skill` returned, as `<bundle>/<skill>`; a call that failed, such as one naming a skill that is not bound, is not counted.

## Requirements

> **Before the release:** until the `uipath-platform`, `uipath` and `uipath-langchain` releases on PyPI ship skills, install your local checkouts in editable mode instead of running `uv sync`. `<uipath-python>` is a checkout of the UiPath Python SDK and `<uipath-langchain-python>` one of this repository:
>
> ```bash
> uv venv -p 3.11 .venv
> uv pip install -p .venv -e "<uipath-langchain-python>[bedrock]" -e <uipath-python>/packages/uipath -e <uipath-python>/packages/uipath-platform -e <uipath-python>/packages/uipath-core
> ```
>
> Then skip `uv venv` and `uv sync` in the steps below and run the rest from the activated `.venv`.

- Python 3.11+
- A UiPath tenant you can sign in to with `uipath auth`

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
cp .env.example .env       # On Windows PowerShell: Copy-Item .env.example .env
uipath auth
uipath init
```

`uipath auth` adds `UIPATH_URL`, `UIPATH_ACCESS_TOKEN`, `UIPATH_TENANT_ID` and `UIPATH_ORGANIZATION_ID` to `.env`. Run `uipath init` after it: init builds the graph through `make_graph()`, which resolves the bundles, and without credentials it falls back to a graph without the `tools` node. `uipath init` leaves the existing `bindings.json` in place.

## Run

```bash
uipath run agent '{"task": "Supplier Acme'\''s invoice INV-1042 is 45 days overdue. What should I do?"}' --trace-file trace.jsonl
```

On Windows PowerShell pass the input as a file:

```powershell
uipath run agent -f input.json --trace-file trace.jsonl
```

### What to expect

The log shows the bundles being resolved before the graph starts:

```
Resolved 2 skill bundle(s) from mock: finance@3 (2 skills), ops@1 (2 skills)
```

The model sees four skills in its system prompt (`finance/invoice-triage`, `finance/vendor-lookup`, `ops/escalation-policy`, `ops/vendor-lookup`), calls `load_skill` with `ops/escalation-policy`, and answers by following its steps. The output carries `"skills_used": ["ops/escalation-policy"]`. The model chooses which skills to load, so another run may load an extra one, such as `finance/vendor-lookup`.

`trace.jsonl` holds one span per line. Look for:

- `Skill bundles resolved`: the source (`mock`), the bundle and skill counts, and one event per resolved bundle with its folder, version and version mode.
- `Skill load - ops/escalation-policy`: the read of the skill's `SKILL.md`, with the bundle version and `skill.invoked_by` = `model`.

Both bundles hold a `vendor-lookup` skill, so the log also warns that both stay available as `finance/vendor-lookup` and `ops/vendor-lookup`.

## Try an override

At deploy time a binding can be overridden per environment. To try one locally, copy the example to `__uipath/uipath.json`, the file `uipath run` reads runtime arguments from:

```bash
mkdir -p __uipath
cp resource-overwrites.example.json __uipath/uipath.json
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force __uipath; Copy-Item resource-overwrites.example.json __uipath/uipath.json
```

It pins `ops` to version 1 instead of `latest`. Run the agent again: the log starts with `Resource overwrites read from` and the file's path, and the `ops` event on the `Skill bundles resolved` span has `skill.version_mode` = `pinned`. Change `"version"` to `"2"` to see a failure: the folder only holds version 1, so `ops` is reported as `version_not_found` on the span and the agent runs with the `finance` skills only. Delete `__uipath/uipath.json` to go back to the declared bindings.

## Switch the source

`UIPATH_SKILLS_SOURCE` in `.env` selects where bundles come from:

- `mock` (this sample): serves `UIPATH_SKILLS_MOCK_ROOT` through the Skills API client, so the run exercises the same download path as Orchestrator.
- `local`: copies `UIPATH_SKILLS_LOCAL_ROOT` (default `./skills`) directly. Set `UIPATH_SKILLS_SOURCE=local` and, if the folder is elsewhere, `UIPATH_SKILLS_LOCAL_ROOT`.
- `orchestrator`: calls the Orchestrator Skills API, once it is live; the bundles must then exist in the `Shared/Skills` folder.
