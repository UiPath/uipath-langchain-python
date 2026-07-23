# Cross-repo trigger: llm-gateway-config → model onboarding

`trigger-model-onboarding.yml` lives here for review but **belongs in the
`UiPath/llm-gateway-config` repo** at `.github/workflows/trigger-model-onboarding.yml`.
This repo (`uipath-langchain-python`) cannot run it — it only owns the target
workflow it dispatches (`model_onboarding.yml`).

## Flow

```
llm-gateway-config PR
  edits agents Model Hub config (alp/stg values.yaml)
        │
        ▼
trigger-model-onboarding.yml  (in llm-gateway-config)
  1. diff BaseRules.<GEO> keys: base SHA vs head SHA
  2. keys present in head but not base = newly ADDED models
  3. per added model: derive get_chat_model `paths` from the id prefix
        │  gh workflow run --repo UiPath/uipath-langchain-python
        ▼
model_onboarding.yml  (in uipath-langchain-python)
  runs the model-onboarding testcase for alpha + staging
```

## What counts as "a model was added"

A new key under
`MODEL_HUB_CONFIGURATION.agents.BaseRules.<GEO>` in either
`…/agents/alp/defaults/values.yaml` or `…/agents/stg/defaults/values.yaml`.
Keys are unioned across both files and deduped, so one model fires onboarding
once (against alpha+staging), not once per file or per geography.

Verified against the live config: the `BaseRules` geographies are `CH` and
`EU`; the extraction yields 43 model keys; a simulated add of
`anthropic.claude-sonnet-5` + `gpt-5-2025-08-07` was isolated correctly and
mapped to the right paths.

## Model id → paths mapping

| id prefix | paths |
|---|---|
| `anthropic.*` | `bedrock_converse,bedrock_invoke,anthropic_sdk` |
| `gpt-*`, `o1/o3/o4-*` | `azure_responses,azure_chat_completions` |
| `gemini-*` | `vertex` |
| bare `claude-*` (e.g. `claude-opus-4-5@…`) | `vertex` (Vertex Anthropic ids) |
| anything else | skipped with a `::warning::` |

Extend the `case` in the workflow as new vendor prefixes appear.

## Required before it works

- **Secret `ONBOARDING_DISPATCH_TOKEN`** in `llm-gateway-config`: a PAT or app
  token with `actions: write` on `UiPath/uipath-langchain-python`. Without it
  the `gh workflow run` dispatch is unauthorized.

## Open decisions for the gateway team

- **PR vs merge:** currently fires on `pull_request` (validate the model before
  merge). To fire only after merge, change the trigger to
  `push: { branches: [develop], paths: [...] }` and diff against the previous
  commit.
- **`files` input** is hardcoded to `image,pdf`. Text-only models should pass
  `files=""` — if the config distinguishes modality, thread it through.
- **prd** is intentionally not triggered (alpha + staging only, per request).
