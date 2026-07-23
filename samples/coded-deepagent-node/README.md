# Coded DeepAgent Node

This sample embeds a coded DeepAgent as one node in a larger LangGraph workflow.
The parent graph accepts a structured `request`, converts it to a human message,
then delegates launch planning to the agent node.

The compiled parent graph is explicitly marked with
`with_uipath_managed_workspace`. This declares the requirement at the runtime
entrypoint, so it remains reliable when the agent node is wrapped in other
LangChain runnables.

## Requirements

- Python 3.11+
- UiPath runtime credentials for `UiPathAzureChatOpenAI`
- Access to the configured OpenAI model, `gpt-5.4`

## Installation

```bash
cd samples/coded-deepagent-node
uv sync
uv run uipath auth
```

Managed workspace hydration is opt-in. Enable it before running this sample:

```bash
export UIPATH_FEATURE_DeepAgentsWorkspaceHydration=true
```

## Usage

```bash
uv run uipath init
uv run uipath run agent "$(cat input.json)"
```

The planner writes `/launch/plan.md` to its managed workspace before returning
the plan to the parent workflow.
