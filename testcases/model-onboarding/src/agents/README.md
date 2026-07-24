# Coded agents for model-onboarding

Each subfolder is a **coded (LangGraph) agent** exercised by the onboarding
testcase against a configurable model + API flavor (from `model_spec` in
`../../input.json`). Agents are registered in [`__init__.py`](__init__.py) via
`AGENT_REGISTRY`; every registered agent runs its payload per code path in the
onboarding grid.

## Adding an agent

1. Create `agents/<name>/agent.py` exposing:
   - `NAME: str` — the registry key
   - `async def run(model, prompt, files) -> str` — returns `"✓"` or `"✗ ..."`
2. Register it in `__init__.py`'s `AGENT_REGISTRY`.
3. Wire it into `main.py` where the payload should run.

## `file_processing`

Reads a single PDF or image and answers a task about it. This is the **coded
equivalent of a low-code UiPath agent** — the low-code source of truth lives in
[`file_processing/lowcode/`](file_processing/lowcode/) (`agent.json` + the
*Analyze Files* built-in tool resource), built and validated in Studio Web /
Agent Builder.

There is no automated low-code → coded eject in the tooling, so
[`file_processing/agent.py`](file_processing/agent.py) is a faithful
re-implementation: same system prompt, same single-file analysis behavior,
expressed against `uipath_langchain`'s multimodal invoke helper. When the low-code
`agent.json` changes, update `agent.py` to match (the system prompt is the main
thing kept in sync).
