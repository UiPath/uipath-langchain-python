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

## `is_tools`

Tests that the model under test can drive an **Integration Service activity
tool** end to end, across the BYO LLM vendor connector flavors: per selected
flavor it binds one tool wired to that flavor's IS activity, forces a call,
executes it through a real IS connection, feeds the result back, and requires
a non-empty final answer. Cells appear as `is_tools/<flavor>`.

Flavors (verified against the live tenant with `uip is resources describe`):
`azure_openai`, `openai`, `openai_v1`, `bedrock_converse`, `vertex`,
`anthropic`. Bedrock's IS connector exposes only a converse activity, so
there is no separate invoke flavor on the IS side.

Configuration comes from `model_spec.is_tools` in `input.json` — `flavors`
(which cells run), `connections` (per-flavor IS connection id overrides), and
`models` (per-flavor vendor model/deployment for the activity payload). The
defaults in [`is_tools/agent.py`](is_tools/agent.py) target the
`llm_gateway_automated_testing` alpha tenant; **running against another org
requires overriding `connections`** or the cells fail legibly.

Note: this agent's runner is `run_flavor(model, flavor, connection_id,
is_model)` — it needs flavor config, so it does not fit the
`(model, prompt, files)` `AGENT_REGISTRY` signature and is imported directly
by `main.py`.

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
